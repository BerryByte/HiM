import torch
from geoopt.manifolds import PoincareBall,PoincareBallExact
from sentence_transformers import SentenceTransformer
from sentence_transformers.models import Pooling, Transformer
import logging
import torch.nn as nn
import math
import os
import torch.nn.functional as F
from typing import Union, Optional, Iterable
from sm_utils import MeanPooling
import random

logger = logging.getLogger(__name__)
        
class HyperbolicCurvature(nn.Module):
    def __init__(self, init_value=1/1):
        super().__init__()
        # Initialize log_c to a fixed value of 1/384 
        raw_init = torch.log(torch.exp(torch.tensor(init_value)) - 1)
        self.log_c = nn.Parameter(raw_init * torch.ones(1))
    
    def forward(self):
        # Softplus ensures positive curvature c > 0
        return F.softplus(self.log_c)
        
class EuclideanToPoincare(nn.Module):
    def __init__(self, c=1/1, normalize=False, max_norm_factor=0.8):
        super().__init__()
        self.normalize = normalize
        self.beta = nn.Parameter(torch.tensor(1.0, dtype=torch.float))
        self.normalize = False
        self.curvature = HyperbolicCurvature(init_value=1/1)
        self.scale_factor = nn.Parameter(torch.tensor(0.01, dtype=torch.float))
        self.manifold = PoincareBall(c=c)  # Initial value, will be updated in forward

    def forward(self, x):
        # Get current curvature value
        current_c = self.curvature()
        self.manifold = PoincareBall(c=current_c.item())  # Use .item() to convert to scalar
        clamped = torch.clamp(x['sentence_embedding'], -8, 8)
        v = clamped * self.scale_factor
        p = euclidean_to_poincare_torch(v, -current_c)  # Here current_c remains tensor for computation
        if random.random() < 0.001:
            poincare_norm = torch.norm(p, dim=-1)
            print(f"Poincar embedding norms: {poincare_norm.mean().item()}")
            print(f"Curvature: {current_c.item()}")
        
        return {'sentence_embedding': p}

    @classmethod
    def load(cls, model_path):
        module = cls()
        state_dict = torch.load(os.path.join(model_path, 'euclidean_to_poincare.pt'))
        module.load_state_dict(state_dict)
        return module

    def state_dict(self):
        """Get optimizer state dict for saving"""
        try:
            return self.optimizer.state_dict()
        except AttributeError:
            # Fallback for AdamW
            return {}

    def load_state_dict(self, state_dict):
        """Load optimizer state dict"""
        try:
            self.optimizer.load_state_dict(state_dict)
        except AttributeError:
            pass

    def save(self, model_path):
        os.makedirs(model_path, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(model_path, 'euclidean_to_poincare.pt'))
        
class HierarchyMamba(SentenceTransformer):
    r"""`HierarchyMamba` is a subclass of `SentenceTransformer` that extends its functionality
    to support hierarchy encoding using the Poincar\'eBall manifold.
    """

    def __init__(
        self,
        model_name_or_path: Optional[str] = None,
        modules: Optional[Iterable[torch.nn.Module]] = None,
        device: Optional[str] = None,
        cache_folder: Optional[str] = None,
        use_auth_token: Union[bool, str, None] = None,
    ):
        super().__init__(model_name_or_path, modules, device, cache_folder, use_auth_token)
        self._register_buffer = {}

    @property
    def embed_dim(self):
        return 384

    @property
    def manifold(self):
        try:
            # Create manifold with current curvature
            if hasattr(self._modules['1'], 'curvature'):
                current_c = self._modules['1'].curvature().item()  # Make sure to get scalar
            else:
                # Fall back to old method
                current_c = self._modules['1'].c.item()
            
            if hasattr(self._register_buffer["manifold"], 'c'):
                # For PoincareBall
                return PoincareBall(c=current_c)
            elif hasattr(self._register_buffer["manifold"], 'k'):
                # For Lorentz
                return Lorentz(k=current_c)
        except Exception as e:
            print(f"Error creating manifold: {e}")
            return self._register_buffer["manifold"]
            
    @classmethod
    def load_pretrained(cls, model, device: Optional[torch.device] = None):
        base=model
        last_module = EuclideanToPoincare(c=1/1, normalize=False)
        tr_model = cls(modules=[base, last_module], device=device)
        tr_model._register_buffer["manifold"] = last_module.manifold
        print(f"Initial curvature parameter: {last_module.curvature().item()}")
        return tr_model

def euclidean_to_poincare_torch(x, curvature=-0.1):
    """
    Project Euclidean vectors to the Poincar ball model with specified curvature.
    
    Parameters:
    x: torch.Tensor - Euclidean vector(s) to project
    curvature: float - the curvature of the hyperbolic space (negative value)
    
    Returns:
    torch.Tensor - the projected vector(s) in the Poincar ball
    """
    # Handle zero vectors
    eps = 1e-15
    norm = torch.norm(x, dim=-1, keepdim=True).clamp(min=eps)
    
    # Compute c and sqrt(c)
    c = -curvature
    #c_sqrt = torch.sqrt(torch.tensor(c))
    c_sqrt = torch.sqrt(c)
    
    # Compute the projection
    factor = torch.tanh(c_sqrt * norm / 2) / (c_sqrt * norm)
    return x * factor