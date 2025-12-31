import os
import wandb
from transformers.integrations import WandbCallback
from types import MethodType

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from M_loader import *
from testing_score import cal_score

from datasets import load_dataset
from sentence_transformers import SentenceTransformer, InputExample, losses, evaluation
from sentence_transformers.models import Transformer, Pooling
from transformers import AutoTokenizer


# 1) Ensure W&B knows your project and logs checkpoints
os.environ["WANDB_PROJECT"] = "MambaSentenceTransformer"
os.environ["WANDB_LOG_MODEL"] = "false"           # log model checkpoints
wandb.login()                                    # prompts for your API key if not set
wandb.init(
    project=os.environ["WANDB_PROJECT"],
    name="mamba_nli_and_stsb_training"
)

# 2) Prepare tokenizer and base Mamba model
# ↓ correct ASCII hyphens everywhere ↓
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

mamba_model = MambaSentenceModel(tokenizer)

# Attach a .tokenize() to embeddings
def tokenize_method(self, texts):
    if isinstance(texts, str):
        texts = [texts]
    return tokenizer(
        texts,
        padding=True,
        truncation=True,
        return_tensors="pt",
        max_length=128
    )

mamba_model.embeddings.tokenize = MethodType(tokenize_method, mamba_model.embeddings)

# Python
class MambaWrapper(nn.Module):
    def __init__(self, mamba_model, tokenizer):
        super().__init__()
        self.mamba = mamba_model
        self.tokenizer = tokenizer

    def forward(self, features):
        return {"sentence_embedding": self.mamba(features)["sentence_embedding"]}

    # python
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
        # Create a minimal __main__.py file
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

model_wrapper = MambaWrapper(mamba_model, tokenizer)
mamba_model = SentenceTransformer(modules=[model_wrapper])
mamba_model.tokenizer = tokenizer

# 4) Optional: watch gradients & parameters
wandb.watch(mamba_model, log="all", log_freq=100)

# 5) Validation evaluator for your custom score
class ValidationEvaluator(evaluation.SentenceEvaluator):
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def __call__(self, model, output_path, epoch, steps):
        print(f"Epoch {epoch} completed. Running validation...")
        val_score = cal_score(model.cuda(), self.tokenizer, 'validation', 128)
        print(f"Validation score: {val_score}")
        # Log to W&B
        wandb.log({"validation_score": val_score, "epoch": epoch})
        return val_score

validation_evaluator = ValidationEvaluator(mamba_model, tokenizer)

# quick sanity-check
print(cal_score(mamba_model.cuda(), tokenizer,'validation',128))

"""
This file contains an example how to make a SentenceTransformer model faster and lighter.

This is achieved by using Knowledge Distillation: We use a well working teacher model to train
a fast and light student model. The student model learns to imitate the produced
sentence embeddings from the teacher. We train this on a diverse set of sentences we got
from SNLI + Multi+NLI + Wikipedia.

After the distillation is finished, the student model produce nearly the same embeddings as the
teacher, however, it will be much faster.

The script implements to options two options to initialize the student:
Option 1: Train a light transformer model like TinyBERT to imitate the teacher
Option 2: We take the teacher model and keep only certain layers, for example, only 4 layers.

Option 2) works usually better, as we keep most of the weights from the teacher. In Option 1, we have to tune all
weights in the student from scratch.

There is a performance - speed trade-off. However, we found that a student with 4 instead of 12 layers keeps about 99.4%
of the teacher performance, while being 2.3 times faster.
"""

import logging
import traceback
from datetime import datetime

import pandas as pd
import torch
from datasets import Dataset, concatenate_datasets, load_dataset
from sklearn.decomposition import PCA

from sentence_transformers import LoggingHandler, SentenceTransformer, evaluation, losses, models
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from sentence_transformers.similarity_functions import SimilarityFunction
from sentence_transformers.trainer import SentenceTransformerTrainer
from sentence_transformers.training_args import SentenceTransformerTrainingArguments

#### Just some code to print debug information to stdout
logging.basicConfig(
    format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO, handlers=[LoggingHandler()]
)
#### /print debug information to stdout


# Teacher Model: Model we want to distill to a smaller model
teacher_model_name = "sentence-transformers/all-MiniLM-L12-v2"
teacher_model = SentenceTransformer(teacher_model_name)

output_dir = "output-1/model-distillation-" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# We will train a small model like TinyBERT to imitate the teacher.
# You can find some small BERT models here: https://huggingface.co/nreimers
student_model_name = "mamba2-17"
student_model = mamba_model

inference_batch_size = 512
train_batch_size = 512

logging.info("Load the AllNLI dataset")
# Load the AllNLI dataset: https://huggingface.co/datasets/sentence-transformers/all-nli
nli_train_dataset = load_dataset("sentence-transformers/all-nli", "pair-score", split="train")
nli_eval_dataset = load_dataset("sentence-transformers/all-nli", "pair-score", split="dev")
# Concatenate all sentences into a new column "sentence"


def combine_sentences(batch):
    return {"sentence": batch["sentence1"] + batch["sentence2"]}


nli_train_dataset = nli_train_dataset.map(
    combine_sentences, batched=True, remove_columns=nli_train_dataset.column_names
)
nli_eval_dataset = nli_eval_dataset.map(combine_sentences, batched=True, remove_columns=nli_eval_dataset.column_names)


def deduplicate(dataset):
    df = pd.DataFrame(dataset)
    df = df.drop_duplicates()
    return Dataset.from_pandas(df, preserve_index=False)


nli_train_dataset = deduplicate(nli_train_dataset)
nli_eval_dataset = deduplicate(nli_eval_dataset)
logging.info(nli_train_dataset)


logging.info("Load the STSB dataset")
# Load the STSB eval/test datasets: https://huggingface.co/datasets/sentence-transformers/stsb
stsb_eval_dataset = load_dataset("sentence-transformers/stsb", split="validation")
stsb_test_dataset = load_dataset("sentence-transformers/stsb", split="test")
logging.info(stsb_eval_dataset)


logging.info("Load the Wikipedia dataset")
# Load the Wikipedia dataset: https://huggingface.co/datasets/sentence-transformers/wikipedia-en-sentences
wikipedia_train_dataset = load_dataset("sentence-transformers/wikipedia-en-sentences", split="train")
# Take 5000 random sentences from the Wikipedia dataset for evaluation
wikipedia_train_dataset_dict = wikipedia_train_dataset.train_test_split(test_size=5000)
wikipedia_train_dataset = wikipedia_train_dataset_dict["train"]
wikipedia_eval_dataset = wikipedia_train_dataset_dict["test"]
logging.info(wikipedia_train_dataset)


# Concatenate the NLI and Wikipedia datasets for training
train_dataset: Dataset = concatenate_datasets([nli_train_dataset, wikipedia_train_dataset])
# Create a relatively small dataset for evaluation
eval_dataset: Dataset = concatenate_datasets(
    [nli_eval_dataset.select(range(5000)), wikipedia_eval_dataset.select(range(5000))]
)

# Create an STSB evaluator
dev_evaluator_stsb = EmbeddingSimilarityEvaluator(
    sentences1=stsb_eval_dataset["sentence1"],
    sentences2=stsb_eval_dataset["sentence2"],
    scores=stsb_eval_dataset["score"],
    main_similarity=SimilarityFunction.COSINE,
    name="sts-dev",
)
logging.info("Teacher Performance")
dev_evaluator_stsb(teacher_model)



# Use the teacher model to get the gold embeddings
def map_embeddings(batch):
    return {
        "label": teacher_model.encode(
            batch["sentence"], batch_size=inference_batch_size, show_progress_bar=False
        ).tolist()
    }


train_dataset = train_dataset.select(range(500000))
train_dataset = train_dataset.map(map_embeddings, batched=True, batch_size=50000)
# Optionally, save the dataset to disk to speed up future runs
#train_dataset.save_to_disk("datasets/distillation_train_dataset")
# from datasets import DatasetDict, load_from_disk

# train_dataset = load_from_disk("datasets/distillation_train_dataset")
# if isinstance(train_dataset, DatasetDict):
#     train_dataset = train_dataset["train"]
eval_dataset = eval_dataset.map(map_embeddings, batched=True, batch_size=50000)

from sentence_transformers.losses import MSELoss
import torch
import torch.nn.functional as F

class DebugNormalizedMSECosineLoss(MSELoss):
    """
    α·MSE(ℓ₂‑normalized embeddings) + (1–α)·(1 – mean_cosine_similarity)
    Debug prints shapes & first‑few values.
    """
    def __init__(self, model, alpha: float = 0.5):
        """
        model: a SentenceTransformer
        alpha: weight on normalized‐MSE vs. cosine term
        """
        super().__init__(model)
        self.alpha = alpha

    def forward(self, sentence_features, labels: torch.Tensor) -> torch.Tensor:
        # ---- 1) DEBUG: inspect inputs ----


        # ---- 2) compute student embeddings ----
        if len(sentence_features) > 1:
            # multiple inputs -> concat on batch dim
            embs = []
            for inputs in sentence_features:
                out = self.model(inputs)["sentence_embedding"]
                embs.append(out)
            student_emb = torch.cat(embs, dim=0)
            # repeat labels for each input set
            teacher_emb = labels.repeat(len(sentence_features), 1).to(student_emb.device)
        else:
            student_emb = self.model(sentence_features[0])["sentence_embedding"]
            teacher_emb = labels.to(student_emb.device)



        # ---- 3) normalize embeddings ----
        s_norm = F.normalize(student_emb, dim=1)
        t_norm = F.normalize(teacher_emb, dim=1)


        # ---- 4) compute losses ----
        loss_mse = self.loss_fct(s_norm, t_norm)
        cos_sim  = (s_norm * t_norm).sum(dim=1).mean()
        loss_cos = 1 - cos_sim
        total    = self.alpha * loss_mse + (1 - self.alpha) * loss_cos

        return total


train_loss = DebugNormalizedMSECosineLoss(model=student_model, alpha=0.5)

# We create an evaluator, that measure the Mean Squared Error (MSE) between the teacher and the student embeddings
eval_sentences = eval_dataset["sentence"]
dev_evaluator_mse = evaluation.MSEEvaluator(eval_sentences, eval_sentences, teacher_model=teacher_model)
dev_evaluator = evaluation.SequentialEvaluator([dev_evaluator_stsb, dev_evaluator_mse])

# Define the training arguments
args = SentenceTransformerTrainingArguments(
    # Required parameter:
    output_dir=output_dir,
    # Optional training parameters:
    num_train_epochs=10,
    per_device_train_batch_size=train_batch_size,
    per_device_eval_batch_size=train_batch_size,
    warmup_ratio=0.1,
    fp16=True,  # Set to False if you get an error that your GPU can't run on FP16
    bf16=False,  # Set to True if you have a GPU that supports BF16
    metric_for_best_model="eval_sts-dev_spearman_cosine",
    load_best_model_at_end=True,
    learning_rate=1e-4,
    # Optional tracking/debugging parameters:
    eval_strategy="steps",
    eval_steps=200,
    save_strategy="steps",
    save_steps=200,
    save_total_limit=2,
    logging_steps=100,
    run_name="distillation-layer-reduction",  # Will be used in W&B if `wandb` is installed
)

# Create the trainer & start training
trainer = SentenceTransformerTrainer(
    model=student_model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    loss=train_loss,
    evaluator=dev_evaluator,
)
trainer.train()

# Evaluate the model performance on the STS Benchmark test dataset
test_evaluator = EmbeddingSimilarityEvaluator(
    sentences1=stsb_test_dataset["sentence1"],
    sentences2=stsb_test_dataset["sentence2"],
    scores=stsb_test_dataset["score"],
    main_similarity=SimilarityFunction.COSINE,
    name="sts-test",
)
test_evaluator(student_model)

# Save the trained & evaluated model locally
final_output_dir = f"{output_dir}/final"
trainer.model.save(final_output_dir)

#load the model and test
# python
student_model = SentenceTransformer(final_output_dir)
student_model.eval()

student_model.tokenizer = tokenizer
student_model.eval()
test_evaluator = EmbeddingSimilarityEvaluator(
    sentences1=stsb_test_dataset["sentence1"],
    sentences2=stsb_test_dataset["sentence2"],
    scores=stsb_test_dataset["score"],
    main_similarity=SimilarityFunction.COSINE,
    name="sts-test",
)
print("Testing the model on STS-B test set")
test_evaluator(student_model)
dev_evaluator_stsb(student_model)
