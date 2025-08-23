import torch
from torch import nn
from torch.optim import AdamW
from pathlib import Path
from tqdm import tqdm
import mlflow
from mlflow import pytorch as mlflow_pytorch
from torch.optim.lr_scheduler import ReduceLROnPlateau
import time
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))

from model.transformer import Transformer 
from data.processed.prepare_dataloaders import get_dataloaders, Tokenizer 
from mlflow_utils.mlflow_helper import setup_mlflow_run


# Model Hyperparameters
MODEL_DIM = 256
NUM_HEADS = 8
NUM_ENCODER_BLOCKS = 4
NUM_DECODER_BLOCKS = 4
CONTEXT_LENGTH = 256 

# Training Hyperparameters
NUM_EPOCHS = 10
LEARNING_RATE = 1e-4
BATCH_SIZE = 16

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

Path("checkpoints").mkdir(exist_ok=True)

# Load Data
print("Loading data...")
train_loader, val_loader, _, ca_tokenizer, en_tokenizer = get_dataloaders(BATCH_SIZE, CONTEXT_LENGTH)
print("Data loaded successfully.")

# Initialize Model, Optimizer, and Loss Function
print("Initializing model...")
model = Transformer(
    vocab_size_encoder=ca_tokenizer.vocab_size,
    vocab_size_decoder=en_tokenizer.vocab_size,
    context_length_encoder=CONTEXT_LENGTH,
    context_length_decoder=CONTEXT_LENGTH,
    model_dim=MODEL_DIM,
    num_blocks=NUM_ENCODER_BLOCKS,
    num_heads=NUM_HEADS
).to(device)

optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
loss_fn = nn.CrossEntropyLoss(ignore_index=en_tokenizer.pad_id)
scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=2)


print(f"Model initialized with {sum(p.numel() for p in model.parameters()):,} parameters.")

# Training and Validation Loops

def train_one_epoch(epoch_index):
    """Performs one full training pass over the training data."""
    model.train()
    running_loss = 0.0
    
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch_index+1}/{NUM_EPOCHS} [Training]")
    
    for i, batch in enumerate(progress_bar):
        encoder_input = batch["encoder_input_ids"]
        decoder_input = batch["decoder_input_ids"]
        src_mask = batch["src_mask"]
        tgt_mask = batch["tgt_mask"]
        labels = batch["labels"]

        # Forward pass
        optimizer.zero_grad() # Clear gradients from the last step
        
        logits = model(encoder_input, decoder_input, src_mask, tgt_mask)
        
        # Loss calculation
        loss = loss_fn(logits.reshape(-1, en_tokenizer.vocab_size), labels.reshape(-1))

        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        progress_bar.set_postfix(loss=running_loss / (i + 1))
        
    return running_loss / len(train_loader)


def validate_one_epoch():
    model.eval()
    running_vloss = 0.0
    
    with torch.no_grad(): # Disable gradient calculation for validation
        progress_bar = tqdm(val_loader, desc="Validating")
        for batch in progress_bar:
            encoder_input = batch["encoder_input_ids"]
            decoder_input = batch["decoder_input_ids"]
            src_mask = batch["src_mask"]
            tgt_mask = batch["tgt_mask"]
            labels = batch["labels"]

            logits = model(encoder_input, decoder_input, src_mask, tgt_mask)
            
            loss = loss_fn(logits.reshape(-1, en_tokenizer.vocab_size), labels.reshape(-1))
            running_vloss += loss.item()
            
            progress_bar.set_postfix(vloss=running_vloss / (len(progress_bar)))
            
    return running_vloss / len(val_loader)


# Main Training Execution
if __name__ == "__main__":
    params = {
        "model_dim": MODEL_DIM,
        "num_heads": NUM_HEADS,
        "num_encoder_blocks": NUM_ENCODER_BLOCKS,
        "num_decoder_blocks": NUM_DECODER_BLOCKS,
        "context_length": CONTEXT_LENGTH,
        "learning_rate": LEARNING_RATE,
        "batch_size": BATCH_SIZE,
        "epochs": NUM_EPOCHS,
        "dataset_subset_size": 100000 # Assuming this is set in prepare_dataloaders.py
    }
    tags = {"model_type": "Transformer", "task": "Translation"}
    run_name = f"transformer_training_{int(time.time())}"
    
    setup_mlflow_run(run_name, params, tags)
    best_vloss = float('inf')
    
    try:
        for epoch in range(NUM_EPOCHS):
            start_time = time.time()
            
            avg_loss = train_one_epoch(epoch)
            avg_vloss = validate_one_epoch()

            scheduler.step(avg_vloss)
            
            end_time = time.time()
            epoch_duration = end_time - start_time
            
            print(f"EPOCH {epoch+1} | Train Loss: {avg_loss:.4f} | Val Loss: {avg_vloss:.4f} | Duration: {epoch_duration:.2f}s")

            # MLFlow Metric Logging
            mlflow.log_metric("train_loss", avg_loss, step=epoch)
            mlflow.log_metric("val_loss", avg_vloss, step=epoch)
            mlflow.log_metric("learning_rate", optimizer.param_groups[0]['lr'], step=epoch)
            
            # Save the model only if validation loss has improved
            if avg_vloss < best_vloss:
                best_vloss = avg_vloss
                save_path = f"checkpoints/transformer_best_model.pth"
                torch.save(model.state_dict(), save_path)
                print(f"New best model saved to {save_path}")

                input_example_batch = next(iter(val_loader))
                input_example = (
                    input_example_batch["encoder_input_ids"], 
                    input_example_batch["decoder_input_ids"], 
                    input_example_batch["src_mask"], 
                    input_example_batch["tgt_mask"]
                )

                # MLFlow Model Logging
                mlflow_pytorch.log_model(
                    pytorch_model=model,
                    artifact_path="model",
                    registered_model_name="Catalanformer-best",
                    input_example=input_example
                )
                print("Best model logged to MLFlow.")
    finally:
            mlflow.end_run()
            print("✅ MLFlow Run Ended.")

    print("Training complete!")

