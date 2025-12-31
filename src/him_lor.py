import torch
import os
from geoopt.manifolds.lorentz import Lorentz
from sentence_transformers import SentenceTransformer
import torch.nn.functional as F
from typing import Union, Optional, Iterable
import logging
import torch.nn as nn
from sm_utils import MeanPooling
import random

logger = logging.getLogger(__name__)

class HyperbolicCurvature(nn.Module):
    def __init__(self, init_value=1/1):
        super().__init__()
        # Initialize log_c to a fixed value of 1/1
        raw_init = torch.log(torch.exp(torch.tensor(init_value)) - 1)
        self.log_c = nn.Parameter(raw_init * torch.ones(1))
    
    def forward(self):
        # Softplus ensures positive curvature c > 0
        return F.softplus(self.log_c)

def project_to_lorentz(v: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """
    Project Euclidean vectors v into the Lorentz model (hyperboloid) with curvature -c.
    v: Tensor of shape (..., n) representing tangent vectors at the hyperbolic origin.
    c: Positive scalar (curvature magnitude).
    Returns: Tensor of shape (..., n+1) representing points on the hyperboloid.
    """
    # Use double precision for calculations
    v = v.to(torch.float64)
    #sqrt_c = torch.sqrt(torch.tensor(c, dtype=torch.float64, device=v.device))
    sqrt_c = torch.sqrt(c.to(torch.float64))
    # Compute norm of v (Euclidean norm in the tangent space)
    norm = torch.norm(v, p=2, dim=-1, keepdim=True)  # shape (..., 1)
    # Handle the zero-vector case to avoid division by zero
    zero_mask = norm.eq(0)  # mask of entries where v is zero
    norm_safe = norm.clone()
    norm_safe[zero_mask] = 1.0  # temporary value to avoid div-by-zero (will not be used)

    # Compute the argument x = sqrt(c) * ||v||
    x = sqrt_c * norm_safe  # shape (..., 1)
    # Compute hyperbolic functions safely
    # Initialize with normal cosh and sinh
    cosh_x = torch.cosh(x)
    sinh_x = torch.sinh(x)
    # Use series expansion for small x to avoid precision loss
    small_x_mask = (x.abs() < 1e-3)
    if small_x_mask.any():
        # For small x, cosh(x) ~ 1 + x^2/2, sinh(x) ~ x + x^3/6
        x_small = x[small_x_mask]
        cosh_approx = 1.0 + 0.5 * (x_small ** 2)
        sinh_approx = x_small + (x_small ** 3) / 6.0
        cosh_x[small_x_mask] = cosh_approx
        sinh_x[small_x_mask] = sinh_approx
    # Clamp large x to avoid overflow (if x > ~50, cosh/sinh may overflow in double)
    large_x_mask = (x > 50.0)
    if large_x_mask.any():
        x_cap = torch.full_like(x[large_x_mask], 50.0)
        cosh_x[large_x_mask] = torch.cosh(x_cap)
        sinh_x[large_x_mask] = torch.sinh(x_cap)

    # Now assemble the Lorentz coordinates
    x0 = cosh_x / sqrt_c  # time component
    spatial = v * (sinh_x / (sqrt_c * norm_safe))  # spatial components
    # For zero norm vectors, define the limit: sinh(x)/x -> 1, so spatial = v (which is 0)
    spatial[zero_mask.expand_as(spatial)] = 0.0  # (origin stays at origin)
    x0[zero_mask] = 1.0 / sqrt_c  # origin's time component = 1/√c

    # Concatenate time and spatial parts
    result = torch.cat([x0, spatial], dim=-1)
    # (Optional) Project onto hyperboloid to correct any tiny numerical error:
    # Minkowski norm might not be exactly -1/c due to rounding; re-normalize:
    # result = renormalize_to_hyperboloid(result, c)  # Pseudocode for clarity

    return result.to(v.dtype)  # cast back to original dtype if needed


class EuclideanToPoincare(nn.Module):
    def __init__(self, c=1/1, normalize=False, max_norm_factor=0.8):
        super().__init__()
        self.normalize = normalize
        self.curvature = HyperbolicCurvature(init_value=1/1)
        self.scale = nn.Parameter(torch.tensor(0.01))
        self.manifold = Lorentz(k=c)

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

    def forward(self, x):
        current_c = self.curvature()
        self.manifold = Lorentz(k=current_c.item())  # Use .item() to convert to scalar
        clamped = torch.clamp(x['sentence_embedding'], -8, 8)
        v = clamped * self.scale
        v = torch.cat([torch.zeros_like(v[..., :1]), v], dim=-1)
        p = project_to_lorentz(v, current_c)  # Here current_c remains tensor for computation
        if random.random() < 0.001:
            lorentz_sqnorm = torch.sum(p * p * torch.tensor([-1.0] + [1.0] * (p.shape[-1]-1), 
                                      device=p.device), dim=-1)
            print(f"Lorentz embedding squared norms: {lorentz_sqnorm.mean().item()}")
            print(f"Curvature: {current_c.item()}")
        
        return {'sentence_embedding': p}

class HierarchyMamba(SentenceTransformer):
    """
    `HierarchyTransformer` is a subclass of `SentenceTransformer` that extends its functionality
    to support hierarchy encoding using the Lorentz manifold.
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
        # Use a dictionary to store the manifold
        self._register_buffer = {}
        self.embed_dim = 384


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