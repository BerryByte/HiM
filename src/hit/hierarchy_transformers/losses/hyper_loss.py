# Copyright 2023 Yuan He
# Modifications 2025 [Modified by anonymous authors for TMLR submission]
# Original source: https://github.com/KRR-Oxford/HierarchyTransformers

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Iterable, Dict, Union, Tuple
import torch
import logging
import math
from .cluster_loss import *
from .centri_loss import *
from .cone_loss import *
from ..models import HierarchyTransformer

logger = logging.getLogger(__name__)

def get_curvature_parameter(model):
    """Safe function to get curvature parameter regardless of model structure."""
    if hasattr(model, 'module'):
        model = model.module
    
    # Check for new curvature module
    if len(model._modules) >= 2 and '1' in model._modules and hasattr(model._modules['1'], 'curvature'):
        return model._modules['1'].curvature().item()
    
    if len(model._modules) >= 2 and '1' in model._modules and hasattr(model._modules['1'], 'c'):
        return model._modules['1'].c.item()
    
    # Otherwise search for c parameter recursively
    for name, module in model.named_modules():
        if hasattr(module, 'curvature'):
            return module.curvature().item()
        if hasattr(module, 'c') and isinstance(module.c, torch.nn.Parameter):
            return module.c.item()
            
    raise ValueError("Could not find curvature parameter in model")
    
class HyperbolicLoss(torch.nn.Module):
    """Hyperbolic loss that combines defined individual losses and applies weights."""

    def __init__(
        self,
        model: HierarchyTransformer,
        apply_triplet_loss: bool = False,
        *weight_and_loss: Tuple[
            float, Union[ClusteringConstrastiveLoss, CentripetalContrastiveLoss, EntailmentConeConstrastiveLoss]
        ],
    ):
        super().__init__()

        self.model = model
        self.apply_triplet_loss = apply_triplet_loss
        self.weight_and_loss = weight_and_loss

    def get_config_dict(self):
        # distance_metric_name = self.distance_metric.__name__
        config = {"distance_metric": f"combined"}
        for weight, loss_func in self.weight_and_loss:
            config[type(loss_func).__name__] = {"weight": weight, **loss_func.get_config_dict()}
        return config
        
    def update_margins(self):
        """Update loss margins based on current curvature"""
        try:
            # Get curvature using robust method
            current_c = get_curvature_parameter(self.model)
            R = 1 / math.sqrt(current_c)
            
            # Update margins for each loss function
            for _, loss_func in self.weight_and_loss:
                if isinstance(loss_func, ClusteringTripletLoss):
                    loss_func.margin = 0.255 * R
                elif isinstance(loss_func, CustomCentripetalTripletLoss):
                    loss_func.margin = 0.0051 * R
                elif isinstance(loss_func, ClusteringConstrastiveLoss):
                    loss_func.negative_margin = 0.255 * R                
            return True
        except Exception as e:
            print(f"Error updating dynamic margins: {e}")
            return False

    def forward(self, sentence_features: Iterable[Dict[str, torch.Tensor]], labels: torch.Tensor):
        # Update margins based on current curvature
        self.update_margins()
        
        # Get model device
        model_device = next(self.model.parameters()).device
        
        # Get embeddings
        reps = [self.model(sentence_feature)["sentence_embedding"] for sentence_feature in sentence_features]
        
        # Move all tensors to same device
        if not self.apply_triplet_loss:
            assert len(reps) == 2
            rep_anchor, rep_other = reps
            rep_anchor = rep_anchor.to(model_device)
            rep_other = rep_other.to(model_device)
            labels = labels.to(model_device)
        else:
            assert len(reps) == 3
            rep_anchor, rep_positive, rep_negative = reps
            rep_anchor = rep_anchor.to(model_device)
            rep_positive = rep_positive.to(model_device)
            rep_negative = rep_negative.to(model_device)
        
        weighted_loss = 0.0
        report = {"weighted": None}
        for weight, loss_func in self.weight_and_loss:
            if not self.apply_triplet_loss:
                cur_loss = loss_func(rep_anchor, rep_other, labels)
            else:
                cur_loss = loss_func(rep_anchor, rep_positive, rep_negative)
            report[type(loss_func).__name__] = round(cur_loss.item(), 6)
            weighted_loss += weight * cur_loss
        report["weighted"] = round(weighted_loss.item(), 6)
            
        logging.info(report)
    
        return weighted_loss
