import torch
import torch.nn as nn
from mamba_ssm import Mamba2
from transformers import AutoTokenizer
import os
from torch import Tensor, nn
import json
from typing import List, Dict, Union, Optional, Any
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class MambaEmbeddings(nn.Module):
    """Embeddings layer for Mamba architecture using MiniLM tokenizer"""

    def __init__(self, vocab_size=30522, hidden_size=384,
                 type_vocab_size=2, dropout_prob=0.1):
        super().__init__()
        # Ensure hidden_size is a multiple of 8 for memory alignment
        self.hidden_size = (hidden_size // 8) * 8

        self.word_embeddings = nn.Embedding(vocab_size, self.hidden_size, padding_idx=0)

        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, features):
        # Handle input when it's a dictionary (from SentenceTransformer)
        if isinstance(features, dict):
            input_ids = features['input_ids']
            attention_mask = features.get('attention_mask', None)
        else:
            input_ids = features
            attention_mask = None

        # Now proceed with your existing embedding logic
        word_embeddings = self.word_embeddings(input_ids)

        # Make sure to return a dictionary with both 'sentence_embedding' and 'attention_mask' keys
        result = {'sentence_embedding': word_embeddings}
        if attention_mask is not None:
            result['attention_mask'] = attention_mask

        return result

class MambaEncoder(nn.Module):
    """Stack of Mamba layers"""

    def __init__(self, num_hidden_layers=6, d_model=384, d_state=64, d_conv=4, expand=2, dropout_prob=0.1):
        super().__init__()
        # Ensure d_model is a multiple of 8
        d_model = (d_model // 8) * 8

        self.layers = nn.ModuleList([
            MambaLayer(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                dropout_prob=dropout_prob
            )
            for _ in range(num_hidden_layers)
        ])

    def forward(self, hidden_states_dict):
        # Extract the hidden states and attention mask from the input dictionary
        hidden_states = hidden_states_dict['sentence_embedding']
        attention_mask = hidden_states_dict.get('attention_mask', None)

        for layer in self.layers:
            hidden_states = layer(hidden_states)

        # Return both hidden_states and attention_mask
        return {"sentence_embedding": hidden_states, "attention_mask": attention_mask}


class MambaLayer(nn.Module):
    """Single Mamba layer with expansion and normalization"""

    def __init__(self, d_model=384, d_state=64, d_conv=4, expand=2, dropout_prob=0.1):
        super().__init__()
        # Ensure d_model is a multiple of 8 for memory alignment
        d_model = (d_model // 8) * 8

        self.norm1 = nn.LayerNorm(d_model, eps=1e-12)

        # For Mamba2, we need to ensure:
        # 1. head_dim is a divisor of d_model
        # 2. d_model * expand / head_dim is a multiple of 8


        # Mamba block
        self.mamba = Mamba2(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            headdim=(d_model*expand)//8,
        )
        # Store parameters for debugging
        self.d_model = d_model
        self.expand = expand
        self.head_dim = (d_model*expand)//8

        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, hidden_states):

        # Pre-norm architecture
        normalized_states = self.norm1(hidden_states)

        # Make sure normalized states are contiguous before passing to Mamba
        normalized_states = normalized_states

        # Apply Mamba
        mamba_output = self.mamba(normalized_states)


        output = hidden_states + self.dropout(mamba_output)

        return output


class MeanPooler(nn.Module):
    """Mean pooling layer that takes average of all token embeddings except special tokens"""

    def __init__(self):
        super().__init__()

    def forward(self, hidden_states, attention_mask=None):
        # Handle input when it's a dictionary
        if isinstance(hidden_states, dict):
            attention_mask = hidden_states.get('attention_mask', attention_mask)
            hidden_states = hidden_states['sentence_embedding']

        # If no attention mask is provided, create a default one (all 1s)
        if attention_mask is None:
            attention_mask = torch.ones(hidden_states.size()[:2],
                                        device=hidden_states.device)

        # Use attention mask to exclude padding tokens
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
        # Sum the embeddings and divide by the sum of the mask
        sum_embeddings = torch.sum(hidden_states * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        mean_embeddings = sum_embeddings / sum_mask
        return mean_embeddings

class MambaSentenceModel(nn.Module):
    """
    Mamba-based sentence transformer with approximately 22M parameters

    Architecture compatible with sentence-transformers/all-MiniLM-L6-v2 tokenizer
    """

    def __init__(self, tokenizer, vocab_size=30522, hidden_size=384, num_hidden_layers=4
                 , type_vocab_size=2, d_state=96,
                 d_conv=4, expand=2, dropout_prob=0.2):
        super().__init__()

        # Ensure hidden_size is a multiple of 8
        hidden_size = (hidden_size // 8) * 8

        self.tokenizer = tokenizer  # Add tokenizer attribute

        self.embeddings = MambaEmbeddings(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            type_vocab_size=type_vocab_size,
            dropout_prob=dropout_prob
        )

        self.encoder = MambaEncoder(
            num_hidden_layers=num_hidden_layers,
            d_model=hidden_size,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            dropout_prob=dropout_prob
        )
        custom_weight_init(self.encoder)

        # Use mean pooling for sentence transformers
        self.pooler = MeanPooler()

    def _find_model_file(self, path: str) -> str:
        """Helper to find the model file in various possible locations"""
        possible_paths = [
            os.path.join(path, "0_SentenceMamba", "pytorch_model.bin"),  # SentenceTransformer format
            os.path.join(path, "pytorch_model.bin"),  # Standalone format
            path  # Direct file path
        ]

        for p in possible_paths:
            if os.path.isfile(p):
                return p

        raise ValueError(f"Could not find model file in {path}")

    def _load_model(self, path: str):
        """Helper method to load model from either a file or directory"""
        try:
            # Find the actual model file
            model_file = self._find_model_file(path)

            # Load state dict
            state_dict = torch.load(model_file, map_location='cpu')

            # Handle different state dict formats
            if isinstance(state_dict, dict):
                if 'state_dict' in state_dict:
                    state_dict = state_dict['state_dict']
                self.load_state_dict(state_dict,strict=False)
            else:
                raise ValueError(f"Unexpected state dict format in {model_file}")

            # Try to load config if it exists
            config_path = os.path.join(os.path.dirname(model_file), 'config.json')
            if os.path.exists(config_path):
                with open(config_path) as f:
                    saved_config = json.load(f)
                    if isinstance(saved_config, dict) and 'sentence_mamba_config' in saved_config:
                        self.config.update(saved_config['sentence_mamba_config'])

        except Exception as e:
            raise ValueError(f"Error loading model from {path}: {str(e)}")

    def save(self, path,safe_serialization=False):
        """
        Saves the model to the given path
        Implements save() method required by SentenceTransformer
        """
        os.makedirs(path, exist_ok=True)

        # Save model state dict
        model_path = os.path.join(path, "pytorch_model.bin")
        if safe_serialization:
            torch.save(self.state_dict(), model_path, _use_new_zipfile_serialization=False)
        else:
            torch.save(self.state_dict(), model_path)

        # Save configuration
        config_path = os.path.join(path, 'config.json')
        with open(config_path, 'w') as f:
            json.dump({
                'model_config': self.config.to_dict()
            }, f, indent=2)

        # Save tokenizer
        self.tokenizer.save_pretrained(path)

    @staticmethod
    def load(path):
        """
        Loads a saved model from the given path
        Implements load() method required by SentenceTransformer
        """
        return MambaSentenceModel()



    def tokenize(self, texts: Union[List[str], List[Dict], List[tuple]]) -> Dict[str, Tensor]:
        """Tokenizes the input texts"""
        return self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128
        )

    def forward(self, features):
        input_ids = features['input_ids']
        attention_mask = features.get('attention_mask', None)
        token_type_ids = features.get('token_type_ids', None)
        position_ids = features.get('position_ids', None)

        # Make sure all input tensors are contiguous
        input_ids = input_ids
        if attention_mask is not None:
            attention_mask = attention_mask
        if token_type_ids is not None:
            token_type_ids = token_type_ids
        if position_ids is not None:
            position_ids = position_ids

        embedding_output = self.embeddings({
            'input_ids': input_ids,
            'token_type_ids': token_type_ids,
            'position_ids': position_ids,
            'attention_mask': attention_mask
        })

        encoder_output = self.encoder(embedding_output)
        sequence_output = encoder_output["sentence_embedding"]

        # Get attention mask from encoder output or use the original one
        attention_mask = encoder_output.get("attention_mask", attention_mask)

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        pooled_output = self.pooler(sequence_output, attention_mask)

        return {"sequence_output": sequence_output, "sentence_embedding": pooled_output}

    def count_parameters(self):
        """Count and print model parameters"""
        total_params = sum(p.numel() for p in self.parameters())
        embedding_params = sum(p.numel() for p in self.embeddings.parameters())
        encoder_params = sum(p.numel() for p in self.encoder.parameters())
        pooler_params = sum(p.numel() for p in self.pooler.parameters())

        print(f"Total parameters: {total_params:,}")
        print(f"Embedding parameters: {embedding_params:,}")
        print(f"Encoder parameters: {encoder_params:,}")
        print(f"Pooler parameters: {pooler_params:,}")

        return total_params

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



# # test the model
# if __name__ == "__main__":
#     # Load tokenizer
#     tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
#
#     # Initialize model
#     model = MambaSentenceModel(tokenizer)
#
#     # Print model summary
#     print(model)
#
#     # Count parameters
#     model.count_parameters()
#
#     # Test tokenization
#     sample_texts = ["Hello, world!", "This is a test sentence."]
#     tokenized_output = model.tokenize(sample_texts)
#     print(tokenized_output)