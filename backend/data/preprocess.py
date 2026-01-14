"""
Data preprocessing: Generate DistilBERT embeddings for comments
"""
import os # File system operations
import numpy as np # Array operations
import pandas as pd # Data manipulation
from transformers import DistilBertTokenizer, DistilBertModel # BERT models
import torch # PyTorch framework
from tqdm import tqdm # Progress bars
from pathlib import Path # Path utilities

DATA_DIR = Path(__file__).parent # Current directory path

def main():
    print("Starting data preprocessing...")

    # Check if raw data exists
    train_path = DATA_DIR / 'train.csv' # Path to CSV

    if not os.path.exists(train_path): # Check file exists
        print(f"Error: {train_path} not found. Please place your dataset there.")
        return

    # Load data
    print("Loading data...")
    df = pd.read_csv(train_path, nrows=50000) # Load first 50K rows
    comments = df['comment_text'].fillna("").tolist() # Extract comments

    # Toxicity labels
    toxicity_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate'] # Label columns
    labels = df[toxicity_cols].values if all(col in df.columns for col in toxicity_cols) else None # Extract labels

    print(f"Processing {len(comments)} comments...")

    # Initialize DistilBERT
    print("Loading DistilBERT model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # Select device
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased') # Load tokenizer
    model = DistilBertModel.from_pretrained('distilbert-base-uncased') # Load model
    model.to(device) # Move to device
    model.eval() # Set evaluation mode

    # Generate embeddings in batches
    batch_size = 32 # Process 32 at once
    embeddings = [] # Store embeddings

    for idx in tqdm(range(0, len(comments), batch_size), desc="Generating embeddings"): # Batch loop
        batch = comments[idx:idx + batch_size] # Get batch

        with torch.no_grad(): # No gradient computation
            inputs = tokenizer( # Tokenize text
                batch, # Current batch
                padding=True, # Pad to max length
                truncation=True, # Truncate long text
                max_length=128, # Max 128 tokens
                return_tensors='pt' # Return PyTorch tensors
            ).to(device) # Move to device

            outputs = model(**inputs) # Forward pass
            # Use [CLS] token embedding (first token)
            batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy() # Extract CLS embeddings
            embeddings.append(batch_embeddings) # Add to list

    embeddings = np.vstack(embeddings) # Stack all embeddings

    # Save processed data
    output_path = DATA_DIR / 'embeddings.npy' # Embeddings output path
    labels_path = DATA_DIR / 'labels.npy' # Labels output path
    texts_path = DATA_DIR / 'comments.txt' # Text output path

    np.save(output_path, embeddings) # Save embeddings
    if labels is not None: # If labels exist
        np.save(labels_path, labels) # Save labels

    with open(texts_path, 'w', encoding='utf-8') as f: # Open text file
        for comment in comments: # Loop through comments
            f.write(comment.replace('\n', ' ') + '\n') # Write one per line

    print(f"\n✓ Preprocessing complete!")
    print(f"  Embeddings shape: {embeddings.shape}")
    print(f"  Saved to: {output_path}")
    if labels is not None: # If labels exist
        print(f"  Toxicity rate: {labels[:, 0].mean():.2%}") # Show toxicity rate

if __name__ == "__main__":
    main() # Run main function
