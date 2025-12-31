import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from deeponto.utils import load_file, set_seed
from setuptools.command.setopt import config_file
from torch.utils.data import DataLoader
import logging
import os
import click
from yacs.config import CfgNode
import math
from hit.hierarchy_transformers.models import *
from him_lor import HierarchyMamba
import json
from hit.hierarchy_transformers.losses import *
from hit.hierarchy_transformers.evaluation import HierarchyTransformerEvaluator
from hit.hierarchy_transformers.utils import prepare_hierarchy_examples, load_hierarchy_dataset, get_torch_device
from itertools import islice
from sentence_transformers import SentenceTransformer
import random
from collections import defaultdict
from datetime import datetime
import torch.distributed as dist
import torch

logger = logging.getLogger(__name__)

def sanitize_path(path_str: str) -> str:
    """Sanitize path string to be filesystem friendly"""
    import re
    # Replace problematic characters
    sanitized = re.sub(r'[\[\],\s]', '_', path_str)
    # Replace multiple underscores with single underscore
    sanitized = re.sub(r'_+', '_', sanitized)
    # Remove any remaining special characters
    sanitized = re.sub(r'[^a-zA-Z0-9\-_/.]', '', sanitized)
    return sanitized

def compute_dynamic_margins(model):
    """Compute clustering and centripetal margins based on current curvature"""
    try:
        # Get curvature using robust method
        curvature = get_curvature_parameter(model)
        R = 1 / math.sqrt(curvature)
        clustering_margin = 0.255 * R
        centripetal_margin = 0.0051 * R
        return clustering_margin, centripetal_margin
    except Exception as e:
        print(f"Error computing dynamic margins: {e}")
        # Fallback to default margins
        return 5.0, 0.1

def get_curvature_parameter(model):
    """Safe function to get curvature parameter regardless of model structure."""
    if hasattr(model, 'module'):
        model = model.module
    
    # Check for new curvature module
    if len(model._modules) >= 2 and '1' in model._modules and hasattr(model._modules['1'], 'curvature'):
        return model._modules['1'].curvature().item()
    
    # Fall back to old method
    if len(model._modules) >= 2 and '1' in model._modules and hasattr(model._modules['1'], 'c'):
        return model._modules['1'].c.item()
    
    # Otherwise search for c parameter recursively
    for name, module in model.named_modules():
        if hasattr(module, 'curvature'):
            return module.curvature().item()
        if hasattr(module, 'c') and isinstance(module.c, torch.nn.Parameter):
            return module.c.item()
            
    raise ValueError("Could not find curvature parameter in model")
      
def check_curvature_grad(model):
    """Check if gradients are flowing to the curvature parameter"""
    if hasattr(model, 'module'):
        model = model.module
    
    # Check for new curvature module first
    if hasattr(model._modules['1'], 'curvature'):
        c_param = model._modules['1'].curvature.log_c
    else:
        c_param = model._modules['1'].c
        
    if c_param.grad is not None:
        print(f"Curvature param grad: {c_param.grad.item()}")
    else:
        print("Curvature parameter has no gradient yet")
        
        
from M_loader import *
class MambaWrapper(nn.Module):
    def __init__(self, mamba_model, tokenizer):
        super().__init__()
        self.mamba = mamba_model
        self.tokenizer = tokenizer

    def forward(self, features):
        return {"sentence_embedding": self.mamba(features)["sentence_embedding"]}
    def save(self, output_path):
        import os, json, torch
        os.makedirs(output_path, exist_ok=True)
        torch.save(self.mamba.state_dict(), os.path.join(output_path, "pytorch_model.bin"))
        with open(os.path.join(output_path, "config.json"), "w") as f:
            json.dump(
                {
                    "model_type": "MambaWrapper",
                    "tokenizer_name_or_path": "sentence-transformers/all-MiniLM-L6-v2"
                },
                f,
                indent=2
            )
        main_path = os.path.join(output_path, "__main__.py")
        with open(main_path, "w") as f:
            f.write("from pretrain.tr_2 import MambaWrapper\n")

    def tokenize(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        return self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=128
        )

    @classmethod
    def load(cls, model_path: str, tokenizer=None, mamba_model_class=None):
        import os, json, torch
        from transformers import AutoTokenizer
        config_path = os.path.join(model_path, "config.json")
        with open(config_path, "r") as f:
            config = json.load(f)
        tokenizer_name = config.get("tokenizer_name_or_path", "sentence-transformers/all-MiniLM-L6-v2")
        tokenizer = tokenizer or AutoTokenizer.from_pretrained(tokenizer_name)
        # Instantiate the base Mamba model (adjust if using a custom class instead)
        mamba_model = MambaSentenceModel(tokenizer)
        model_file = os.path.join(model_path, "pytorch_model.bin")
        state_dict = torch.load(model_file, map_location="cpu")
        mamba_model.load_state_dict(state_dict)
        return cls(mamba_model, tokenizer)

def main(config_file: str, gpu_id: int):
    if "LOCAL_RANK" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
        print(f"DDP initialized: world_size={dist.get_world_size()}, rank={dist.get_rank()}, local_rank={local_rank}")
    else:
        print("Running in non-distributed mode")
    
    if dist.is_initialized():
        print(f"DDP properly initialized: world_size={dist.get_world_size()}, rank={dist.get_rank()}")
    else:
        print("WARNING: DDP not initialized!")
    
    set_seed(random.randint(1,100))
    mult = 5
    config = CfgNode(load_file(config_file))

    # load dataset
    data_path = config.data_path
    dataset, entity_lexicon = load_hierarchy_dataset(data_path)
     # only use train and val for testing
    train_examples = prepare_hierarchy_examples(
        entity_lexicon, dataset["train"], config.apply_hard_negatives, config.apply_triplet_loss
    )
    #train_examples = train_examples[:25000*mult]
    # train_examples = train_examples[:100]
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=256)

    val_examples = prepare_hierarchy_examples(entity_lexicon, dataset["val"], config.apply_hard_negatives)
    test_examples = prepare_hierarchy_examples(entity_lexicon, dataset["test"], config.apply_hard_negatives)
    #val_examples = val_examples[:10000]
    #test_examples = test_examples[:10000]

    device = get_torch_device(gpu_id)
    import torch.nn.init as init
    
    model = SentenceTransformer(
        'output/model-distillation-2025-05-09_03-22-58/final')
    

    # Load the pretrained model
    model = HierarchyMamba.load_pretrained(model, device)

    def random_initialize_model(model):
        # Store the original curvature value
        if hasattr(model[1], 'curvature'):
            original_curvature = model[1].curvature().item()
        else:
            original_curvature = model[1].c.item()
        
        for name, param in model.named_parameters():
            if name == '1.c' or 'curvature' in name or 'log_c' in name:
                continue
                
            if 'weight' in name:
                if param.data.dim() >= 2:
                    init.kaiming_normal_(param.data)
                else:
                    init.normal_(param.data)
            elif 'bias' in name:
                init.constant_(param.data, 0)
        
        # Verify curvature is unchanged
        if hasattr(model[1], 'curvature'):
            final_curvature = model[1].curvature().item()
        else:
            final_curvature = model[1].c.item()
            
        print(f"Curvature before initialization: {original_curvature}")
        print(f"Curvature after initialization: {final_curvature}")

    random_initialize_model(model)
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters())
    print(f'The model has {count_parameters(model):,} trainable parameters')
    
    class CurvatureMonitorCallback:
        def __init__(self, model):
            self.model = model
            self.best_score = -float('inf')
            self.best_c = None
            self.c_history = []
            
        def __call__(self, score, epoch, steps):
            try:
                current_c = get_curvature_parameter(self.model)
                self.c_history.append((epoch, steps, current_c, score))
                print(f"Epoch {epoch}, Steps {steps}, Current c: {current_c}, Score: {score}")
                
                if score > self.best_score:
                    self.best_score = score
                    self.best_c = current_c
                    print(f"New best curvature value: {self.best_c} with score: {self.best_score}")
            except Exception as e:
                print(f"Error monitoring curvature: {e}")
                
            return self.on_evaluation_end(score, epoch, steps)
            
        def on_evaluation_end(self, score, epoch, steps):
            return score
    
    curvature_callback = CurvatureMonitorCallback(model)
    
    class GradientCheckCallback:
        def __init__(self, model, check_every=100):
            self.model = model
            self.check_every = check_every
            self.step_counter = 0
            
        def __call__(self, score, epoch, steps):
            self.step_counter += 1
            if self.step_counter % self.check_every == 0:
                check_curvature_grad(self.model)
            return score
            
    gradient_callback = GradientCheckCallback(model)
    
    class StabilizeCallback:
        def __init__(self, model, stabilize_every=100):
            self.model = model
            self.stabilize_every = stabilize_every
            self.step_counter = 0
            
        def __call__(self, score, epoch, steps):
            self.step_counter += 1
            if self.step_counter % self.stabilize_every == 0:
                self.stabilize_model()
            return score
            
        
        def stabilize_model(self):
            """Project embeddings back to manifold"""
            print("Stabilizing model parameters...")
            if hasattr(self.model, 'module'):
                model = self.model.module
            else:
                model = self.model
            
            device = next(model.parameters()).device
            
            # Get current curvature value
            if hasattr(model._modules['1'], 'curvature'):
                current_c = model._modules['1'].curvature().item()
            else:
                current_c = model._modules['1'].c.item()
                
            # Create a new manifold instance with the current curvature
            if hasattr(model.manifold, 'k'):
                # For Lorentz model
                model._register_buffer["manifold"] = Lorentz(k=current_c)
            elif hasattr(model.manifold, 'c'):
                # For Poincar model
                model._register_buffer["manifold"] = PoincareBall(c=current_c)
            
            # For each parameter in the last layer that requires grad
            for name, param in model._modules['1'].named_parameters():
                if param.requires_grad:
                    # Skip curvature-related parameters
                    if name not in ['c', 'log_c'] and not 'curvature' in name:
                        # Project parameter
                        with torch.no_grad():
                            param.copy_(model.manifold.projx(param))
    
    stabilize_callback = StabilizeCallback(model)

    # loss
    losses = []
    
    cluster_margin, centri_margin = compute_dynamic_margins(model)
    if hasattr(model[1], 'curvature'):
        initial_c = model[1].curvature().item()
    else:
        initial_c = model[1].c.item()
    print(f"Initial curvature: {initial_c:.6f}")
    
    if config.loss.cluster.weight > 0.0:
        if config.apply_triplet_loss:
            cluster_loss = ClusteringTripletLoss(model.manifold, cluster_margin)  # Use dynamic margin
        else:
            cluster_loss = ClusteringConstrastiveLoss(
                model.manifold, config.loss.cluster.positive_margin, cluster_margin  # Use dynamic margin
            )
        losses.append((config.loss.cluster.weight, cluster_loss))
    
    if config.loss.centri.weight > 0.0:
        centri_loss_class = CustomCentripetalTripletLoss
        centri_loss = centri_loss_class(model.manifold, centri_margin)  # Use dynamic margin
        losses.append((config.loss.centri.weight, centri_loss))

    data_suffix = config.data_path.split(os.path.sep)[-1]
    
    hit_evaluator = HierarchyTransformerEvaluator(
        device=device,
        eval_batch_size=512,
        val_examples=val_examples,
        test_examples=test_examples,
        train_examples=train_examples if config.eval_train else None,
    )
    
    output_path = sanitize_path(
        f"experiments/HiML-{config.pretrained}-{data_suffix}"
        f"-hard_{config.apply_hard_negatives}"
        f"-triplet_{config.apply_triplet_loss}"
    )
    
    if dist.is_initialized():
        local_rank = dist.get_rank()
        print(f"Process {local_rank} using device {device}")
    
    hyper_loss = HyperbolicLoss(model, config.apply_triplet_loss, *losses)

    print(hyper_loss.get_config_dict())
    hyper_loss.to(device)

    model.fit(
        train_objectives=[(train_dataloader, hyper_loss)],
        epochs=10,
        optimizer_params={"lr": 1e-4},
        warmup_steps=100,
        evaluator=hit_evaluator,
        weight_decay=1e-3,
        output_path=output_path,
        max_grad_norm=1.0,
        callback=lambda score, epoch, steps: 
            curvature_callback(score, epoch, steps) and 
            gradient_callback(score, epoch, steps) and 
            stabilize_callback(score, epoch, steps)
    )

    final_eval_score = hit_evaluator(model, output_path)
    print(f"Final evaluation score: {final_eval_score}")
    
    
if __name__ == "__main__":

    main("example_config.yaml", 0)
