from mamba2_torch.modeling.modeling_mamba2 import *
from transformers import AutoTokenizer

from mamba2_torch.modeling.configuration_mamba2 import Mamba2Config
from typing import Optional
import os
from transformers import PreTrainedModel

from geoopt.manifolds import PoincareBall
import logging
from typing import Union, Optional, Iterable

class Mamba2Config(Mamba2Config):
    def __init__(
            self,
            vocab_size=30522,
            hidden_size=384,
            intermediate_size=1024,
            num_hidden_layers=4,
            state_size=128,
            num_heads=4,
            head_dim=256,
            conv_kernel=4,
            use_bias=True,
            use_conv_bias=True,
            layer_norm_epsilon=1e-5,
            emb_initializer_range=0.02,
            conv_initializer_range=None,
            rescale_prenorm_residual=False,
            residual_in_fp32=True,
            chunk_size=256,
            A_initializer_range=(1, 16),
            time_step_min=0.001,
            time_step_max=0.1,
            time_step_floor=1e-4,
            output_last_ssm_states=True,
            time_step_limit=(0.0, float("inf")),
            use_triton_kernels=False,
            dropout_rate=0.1,
            use_cache=True,
            hidden_act="silu",  # Add this line
            **kwargs
    ):
        super().__init__(**kwargs)
        self.dropout_rate = dropout_rate
        self.use_cache = use_cache
        self.output_last_ssm_states = output_last_ssm_states
        self.time_step_min = time_step_min
        self.time_step_max = time_step_max
        self.A_initializer_range = A_initializer_range
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.state_size = state_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.conv_kernel = conv_kernel
        self.use_bias = use_bias
        self.use_conv_bias = use_conv_bias
        self.layer_norm_epsilon = layer_norm_epsilon
        self.emb_initializer_range = emb_initializer_range
        self.conv_initializer_range = conv_initializer_range
        self.rescale_prenorm_residual = rescale_prenorm_residual
        self.residual_in_fp32 = residual_in_fp32
        self.time_step_limit = time_step_limit
        self.time_step_floor = time_step_floor
        self.chunk_size = chunk_size
        self.use_triton_kernels = use_triton_kernels
        self.hidden_act = hidden_act  # Add this line


class Mamba2Model(torch.nn.Module):
    def __init__(self):
        super(Mamba2Model, self).__init__()
        self.model, self.tokenizer = self.create_model()
        self.total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.config = self.model.config
        self._embed_dim = self.config.hidden_size

    def save(self, save_path,safe_serialization=False):
        if safe_serialization:
            torch.save(self.state_dict(), save_path)
        else:
            torch.save(self, save_path)

    @property
    def embed_dim(self):
        return self._embed_dim

    def tokenize(self, sentences):
        return self.tokenizer(sentences, return_tensors="pt", padding=True, truncation=True, max_length=512)

    @property
    def manifold(self):
        return self._register_buffer["manifold"]


    def create_model(self):
        # Create the configuration
        config = Mamba2Config()

        # Create the model with language modeling head
        model = Mamba2ForCausalLM(config)

        # Create a matching tokenizer
        #tokenizer = AutoTokenizer.from_pretrained("state-spaces/mamba-130m-hf")
        # Add Dropout layers after each layer in all Mamba2Block instances
        custom_weight_init(model)
        from transformers import AutoTokenizer, AutoModel

        tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L12-v2")
        return model, tokenizer

    def distillation_loss(self, student_logits, teacher_logits, temperature=2.0, alpha=0.5):
        """
        Calculate the distillation loss between student and teacher logits.

        Args:
            student_logits: tensor of shape [batch_size, sequence_length, vocab_size]
            teacher_logits: tensor of shape [batch_size, sequence_length, vocab_size]
            temperature: distillation temperature (default: 2.0)
            alpha: weight for distillation loss (default: 0.5)

        Returns:
            Distillation loss
        """
        # Flatten the sequence dimension
        batch_size, seq_length, vocab_size = student_logits.shape
        student_logits = student_logits.view(-1, vocab_size)
        teacher_logits = teacher_logits.view(-1, vocab_size)

        # Apply temperature scaling
        student_logits_temp = student_logits / temperature
        teacher_logits_temp = teacher_logits / temperature

        # Compute soft targets (teacher probabilities)
        teacher_probs = torch.nn.functional.softmax(teacher_logits_temp, dim=-1)

        # Compute log probabilities of student predictions
        log_student_probs = torch.nn.functional.log_softmax(student_logits_temp, dim=-1)

        # Calculate KL divergence loss
        # We use reduction='batchmean' to properly normalize the loss
        kl_div = torch.nn.functional.kl_div(
            input=log_student_probs,
            target=teacher_probs,
            reduction='batchmean'
        ) * (temperature ** 2)

        return kl_div

    def forward(self, input_ids, attention_mask=None, labels=None, teacher=None, **kwargs):

        return self.model(input_ids, attention_mask=attention_mask, labels=labels, **kwargs)

    def generate(self, *args, **kwargs):
        input_ids = kwargs.pop('input_ids')
        attention_mask = kwargs.pop('attention_mask')
        input_ids = input_ids * attention_mask
        kwargs['input_ids'] = input_ids
        return self.model.generate(*args, **kwargs)




class Mamba2PreTrainedModel(PreTrainedModel):
    config_class = Mamba2Config
    base_model_prefix = "mamba2"
    supports_gradient_checkpointing = True

    def __init__(self, config,teacher=None):
        super().__init__(config)
        self.model = Mamba2Model()
        self.tokenizer = self.model.tokenizer
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)
        self.teacher = None

    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        return self.model(input_ids, attention_mask=attention_mask, labels=labels,teacher=self.teacher, **kwargs)

    def generate(self, *args, **kwargs):
        return self.model.generate(*args, **kwargs)

    def load_model_params(self, load_path):
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"No file found at {load_path}")

        state_dict = torch.load(load_path)
        self.load_state_dict(state_dict)

        print(f"Model parameters loaded from {load_path}")

    def save_model_params(self, save_path):
        torch.save(self.state_dict(), save_path)

        print(f"Model parameters saved to {save_path}")


def custom_weight_init(model):
    def init_weights(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Initialize linear and embedding layers
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            # Initialize LayerNorm layers
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, nn.Conv1d):
            # Initialize Conv1d layers
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                module.bias.data.zero_()

    # First apply general initialization
    model.apply(init_weights)

    # Special initialization for Mamba-specific parameters
    for name, module in model.named_modules():
        if hasattr(module, 'A_log'):
            # Initialize A_log with a stable range
            with torch.no_grad():
                module.A_log.data.uniform_(-1, 1)  # Use a more stable range

        if hasattr(module, 'D'):
            # Initialize D with small positive values
            with torch.no_grad():
                module.D.data.uniform_(0.001, 0.1)

        # Initialize dt_proj parameters if they exist
        if hasattr(module, 'dt_proj'):
            if hasattr(module.dt_proj, 'weight'):
                with torch.no_grad():
                    nn.init.normal_(module.dt_proj.weight, mean=0.0, std=0.02)
            if hasattr(module.dt_proj, 'bias'):
                with torch.no_grad():
                    module.dt_proj.bias.data.zero_()

    # Verification step
    for name, param in model.named_parameters():
        if param.requires_grad:
            # Replace NaN/Inf values with small random values if they exist
            with torch.no_grad():
                nan_mask = torch.isnan(param)
                inf_mask = torch.isinf(param)
                if nan_mask.any() or inf_mask.any():
                    problem_indices = nan_mask | inf_mask
                    param.data[problem_indices] = torch.randn_like(
                        param.data[problem_indices]
                    ) * 0.02

            # Final verification
            assert not torch.isnan(param).any(), f"NaN values still present in {name}"
            assert not torch.isinf(param).any(), f"Inf values still present in {name}"


def check_model_initialization(model):
    """Enhanced model initialization checking function"""
    all_ok = True
    problematic_params = []

    for name, param in model.named_parameters():
        if param.requires_grad:
            stats = {
                "mean": param.data.mean().item(),
                "std": param.data.std().item(),
                "min": param.data.min().item(),
                "max": param.data.max().item(),
                "has_nan": torch.isnan(param.data).any().item(),
                "has_inf": torch.isinf(param.data).any().item()
            }

            if stats["has_nan"] or stats["has_inf"] or abs(stats["mean"]) > 100 or stats["std"] > 100:
                all_ok = False
                problematic_params.append((name, stats))

            # print(f"{name}:")
            # print(f"  Mean: {stats['mean']:.6f}")
            # print(f"  Std: {stats['std']:.6f}")
            # print(f"  Min: {stats['min']:.6f}")
            # print(f"  Max: {stats['max']:.6f}")

    if not all_ok:
        print("\nProblematic parameters found:")
        for name, stats in problematic_params:
            print(f"\n{name}:")
            for key, value in stats.items():
                print(f"  {key}: {value}")

    return all_ok