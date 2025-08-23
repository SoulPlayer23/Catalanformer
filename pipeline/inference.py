import torch
from pathlib import Path
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))
from model.transformer import Transformer 
from data.processed.prepare_dataloaders import get_dataloaders, Tokenizer 

# Configuration
MODEL_DIM = 256
NUM_HEADS = 8
NUM_ENCODER_BLOCKS = 4
NUM_DECODER_BLOCKS = 4
CONTEXT_LENGTH = 256 
BATCH_SIZE = 16

# Load Tokenizers and Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load the tokenizers that were created during training
print("Loading data and tokenizers...")
# We run get_dataloaders to get the fitted tokenizers
_, _, _, ca_tokenizer, en_tokenizer = get_dataloaders(BATCH_SIZE, CONTEXT_LENGTH)

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

# Load the best model weights you saved during training
model_path = Path("checkpoints/transformer_best_model.pth")
if not model_path.exists():
    raise FileNotFoundError(f"Model checkpoint not found at {model_path}")

print(f"Loading model weights from {model_path}...")
model.load_state_dict(torch.load(model_path))
model.eval()

# The Translate Function (Greedy Decoding)
def translate(catalan_sentence: str):
    """
    Translates a Catalan sentence to English using the trained Transformer model.
    """
    # Tokenize and add EOS
    src_tokens = [token.text.lower() for token in ca_tokenizer.spacy.tokenizer(catalan_sentence)]
    src_sequence = [ca_tokenizer.word2idx.get(t, ca_tokenizer.word2idx['<unk>']) for t in src_tokens]
    src_sequence.append(ca_tokenizer.eos_id)
    
    # Convert to tensor and add batch dimension
    encoder_input = torch.tensor(src_sequence, device=device).unsqueeze(0)
    
    # Create the source mask
    src_mask = (encoder_input != ca_tokenizer.pad_id).unsqueeze(1).unsqueeze(2)

    # Get the encoder output
    with torch.no_grad():
        encoder_output = model.encoder(encoder_input, src_mask)

    # Greedy Decoding Loop
    # Start the decoder input with the <sos> token
    decoder_input = torch.tensor([[en_tokenizer.sos_id]], device=device)
    
    for _ in range(CONTEXT_LENGTH):
        # Create the target mask
        tgt_mask = (decoder_input != en_tokenizer.pad_id).unsqueeze(1).unsqueeze(2)
        look_ahead_mask = torch.tril(torch.ones((decoder_input.size(1), decoder_input.size(1)), device=device)).bool()
        tgt_mask = tgt_mask & look_ahead_mask

        # Get model prediction for the next token
        with torch.no_grad():
            logits = model.decoder(decoder_input, encoder_output, src_mask, tgt_mask)
        
        # Get the most likely token (greedy choice)
        next_token = logits[:, -1, :].argmax(dim=-1).item()
        
        # Append the new token to the decoder input
        decoder_input = torch.cat(
            [decoder_input, torch.tensor([[next_token]], device=device)],
            dim=1
        )
        
        # If <eos> is predicted, the sentence is finished
        if next_token == en_tokenizer.eos_id:
            break
            
    # Convert token IDs back to words
    translated_tokens = [en_tokenizer.idx2word[idx] for idx in decoder_input.squeeze(0).tolist()]
    
    # Clean up the output by removing special tokens
    clean_tokens = [t for t in translated_tokens if t not in ['<sos>', '<eos>', '<pad>']]
    
    return " ".join(clean_tokens)

if __name__ == "__main__":
    print("\n--- Catalan to English Translation ---")
    print("Type a Catalan sentence or 'quit' to exit.")
    
    while True:
        catalan_sentence = input("> ")
        if catalan_sentence.lower() == 'quit':
            break
        
        english_translation = translate(catalan_sentence)
        print(f"English: {english_translation}")
