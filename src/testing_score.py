from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from torch.nn.functional import cosine_similarity
from scipy.stats import spearmanr


def cal_score(model, tokenizer,part, batch_size=1024):
    # Load the stsb dataset directly from sentence-transformers
    dataset = load_dataset("sentence-transformers/stsb")
    # Prepare the data
    sentences1 = dataset[part]["sentence1"]
    sentences2 = dataset[part]["sentence2"]
    scores = dataset[part]["score"]

    # Encode the sentences and get predictions
    embeddings1 = []
    embeddings2 = []
    model.eval()

    for i in range(0, len(sentences1), batch_size):
        batch_sentences1 = sentences1[i:i+batch_size]
        batch_sentences2 = sentences2[i:i+batch_size]

        # Tokenize the sentences
        inputs1 = tokenizer(batch_sentences1, return_tensors="pt", padding=True, truncation=True).to('cuda')
        inputs2 = tokenizer(batch_sentences2, return_tensors="pt", padding=True, truncation=True).to('cuda')

        # Get model output
        with torch.no_grad():
            outputs1 = model(inputs1)
            outputs2 = model(inputs2)

            # Extract embeddings and move to CPU
            embedding1_t = outputs1['sentence_embedding'].cpu()
            embedding2_t = outputs2['sentence_embedding'].cpu()

        embeddings1.append(embedding1_t)
        embeddings2.append(embedding2_t)

    # Evaluate
    # Convert embeddings lists to tensors for cosine_similarity
    embeddings1 = torch.cat(embeddings1)
    embeddings2 = torch.cat(embeddings2)

    predictions = cosine_similarity(embeddings1, embeddings2)
    # Flatten the predictions to a 1D array
    predictions = predictions.flatten()
    correlation, _ = spearmanr(predictions, scores)
    return correlation

