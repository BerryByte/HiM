from typing import List, Dict, Union, Optional, Any
from torch import Tensor, nn
import torch
import os
import json
import logging
from sentence_transformers import SentenceTransformer
from sentence_transformers.models import Pooling, Normalize
from class_file import *
import torch.nn.functional as F

max_length = 256
class SentenceMamba(nn.Module):
    """
    Transformer component that wraps the Mamba backbone
    """

    def __init__(
            self,
            model_name_or_path: Optional[str] = None,
            max_seq_length: int = max_length,
            do_lower_case: bool = False
    ):
        super().__init__()

        # Create base mamba model
        base_model = Mamba2Model()
        self.mamba_model = Mamba2PreTrainedModel(base_model.config)
        self.tokenizer = base_model.tokenizer

        # Set initial config
        self.config = {
            'max_seq_length': max_seq_length,
            'do_lower_case': do_lower_case
        }

        # Load model if path is provided
        if model_name_or_path:
            self._load_model(model_name_or_path)

        # Remove language modeling head as we don't need it
        if hasattr(self.mamba_model.model.model, 'lm_head'):
            del self.mamba_model.model.model.lm_head

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
                self.mamba_model.load_state_dict(state_dict,strict=False)
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

    def forward(self, features):
        """Forward pass through the model"""
        features = {key: tensor.to(self.mamba_model.device) for key, tensor in features.items()}
        output = self.mamba_model.model.model.backbone(**features, last_hidden_state=True)

        return {
            'token_embeddings': output['last_hidden_state'],
            'attention_mask': features['attention_mask']
        }

    def get_word_embedding_dimension(self):
        """Returns the dimension of the word embeddings"""
        return self.mamba_model.config.hidden_size

    def tokenize(self, texts: Union[List[str], List[Dict], List[tuple]]) -> Dict[str, Tensor]:
        """Tokenizes the input texts"""
        return self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length
        )

    def save(self, path,safe_serialization=False):
        """
        Saves the model to the given path
        Implements save() method required by SentenceTransformer
        """
        os.makedirs(path, exist_ok=True)

        # Save model state dict
        model_path = os.path.join(path, "pytorch_model.bin")
        if safe_serialization:
            torch.save(self.mamba_model.state_dict(), model_path, _use_new_zipfile_serialization=False)
        else:
            torch.save(self.mamba_model.state_dict(), model_path)

        # Save configuration
        config_path = os.path.join(path, 'config.json')
        with open(config_path, 'w') as f:
            json.dump({
                'sentence_mamba_config': self.config,
                'model_config': self.mamba_model.config.to_dict()
            }, f, indent=2)

        # Save tokenizer
        self.tokenizer.save_pretrained(path)

    @staticmethod
    def load(path):
        """
        Loads a saved model from the given path
        Implements load() method required by SentenceTransformer
        """
        return SentenceMamba(model_name_or_path=path)

    def to(self, device):
        """Move the model to the specified device"""
        self.mamba_model = self.mamba_model.to(device)
        return self


class TanhNormalize(nn.Module):
    """Applies tanh normalization to the input embeddings"""

    def __init__(self):
        super().__init__()


    def forward(self, features: Dict[str, Tensor]) -> Dict[str, Tensor]:
        embeddings = features['sentence_embedding']
        #add a linear layer to the embeddings


        embeddings = torch.tanh(embeddings)  # Apply tanh activation
        embeddings = F.normalize(embeddings, p=2, dim=1)
        features['sentence_embedding'] = embeddings
        return features

    def save(self, path):
        """
        Save function required by SentenceTransformer
        Since this module has no parameters, we just save the config
        """
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, 'config.json'), 'w') as f:
            json.dump({'type': 'TanhNormalize'}, f)

    @staticmethod
    def load(path):
        """Load function required by SentenceTransformer"""
        return TanhNormalize()

def create_sentence_mamba(model_path: Optional[str] = None) -> SentenceTransformer:
    """Creates a SentenceMamba model structured like all-MiniLM-L12-v2"""

    # Create the three main components
    transformer = SentenceMamba(model_path, max_seq_length=max_length)
    pooling = Pooling(
        word_embedding_dimension=768,
        pooling_mode_mean_tokens=True
    )
    tanh_normalize = TanhNormalize()

    # Create SentenceTransformer model
    model = SentenceTransformer(modules=[
        transformer,
        pooling,
        tanh_normalize
    ])

    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    return model

import torch
from torch import nn
import torch.nn.functional as F
from typing import Dict

class MeanPooling(nn.Module):
    def __init__(self, word_embedding_dimension: int = 384):
        super(MeanPooling, self).__init__()
        self.word_embedding_dimension = word_embedding_dimension

    def ensure_scaled_tanh(self,embedding: torch.Tensor) -> torch.Tensor:
        
        # Check if values are already in (-1, 1). If not, apply tanh.
        if not torch.all((embedding > -1) & (embedding < 1)):
            embedding = torch.tanh(embedding)
        
        # Scale the embedding so that the maximum possible norm becomes s * sqrt(d) = 1/sqrt(c)
        return embedding

    def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        token_embeddings = features['token_embeddings']
        attention_mask = features['attention_mask']

        token_embeddings = self.ensure_scaled_tanh(token_embeddings)

        # Mean pooling
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        output_vector = sum_embeddings / sum_mask

        features['sentence_embedding'] = output_vector
        return features

    def get_sentence_embedding_dimension(self) -> int:
        return self.word_embedding_dimension

    def get_config_dict(self) -> Dict[str, int]:
        return {'word_embedding_dimension': self.word_embedding_dimension}

    def save(self, output_path: str) -> None:
        with open(os.path.join(output_path, 'config.json'), 'w') as fOut:
            json.dump(self.get_config_dict(), fOut, indent=2)

    @staticmethod
    def load(input_path: str) -> 'TanhMeanPooling':
        with open(os.path.join(input_path, 'config.json')) as fIn:
            config = json.load(fIn)
        return MeanPooling(**config)



# # Or load a pretrained model
# model = create_sentence_mamba("/home/dettrax/PycharmProjects/HierarchyTransformers/scripts/model_step_17000.pt")

# # Generate embeddings
# sentences = ["This is a test sentence.", "Another example sentence."]
# tokenized_sentences = model.tokenize(sentences)
# embeddings = model(tokenized_sentences.to('cuda'))
#
# model.save_pretrained("/home/dettrax/PycharmProjects/HierarchyTransformers/scripts/mamba-sentence-23")  # Save the model
#
# #load the model
# loaded_model = SentenceMamba.load("/home/dettrax/PycharmProjects/HierarchyTransformers/scripts/mamba-sentence-23/")  # Load the model
#
# # Recreate the SentenceTransformer with the loaded model
# loaded_model = SentenceTransformer(modules=[
#     loaded_model,
#     Pooling(word_embedding_dimension=384, pooling_mode_mean_tokens=True),
#     TanhNormalize()
# ])
#
# loaded_embeddings = loaded_model(tokenized_sentences.to('cuda'))
#
# print(loaded_embeddings.keys())  # This will print the loaded embeddings
# print(embeddings.keys())  # This will print the generated embeddings
#
#