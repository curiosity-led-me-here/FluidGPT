import numpy as np
import tiktoken
import torch
import torch.optim as optim
from model import GPT
import torch.nn.functional as F

# ===== Load data =====
with open("data.txt") as file:
    data = file.read()

encoding = tiktoken.encoding_for_model("gpt-4")
encoded_data = encoding.encode(data)
print("Token count:", len(encoded_data))

# ===== Create input-target pairs =====
Y = torch.tensor(encoded_data[1:], dtype=torch.long)
X = torch.tensor(encoded_data[:-1], dtype=torch.long)

# ===== Hyperparameters =====
batch_size = 100
block_size = 32
embed_dim = 128
num_heads = 4
FFN_depth = 512
encoder_layers = 6
epochs = 100  # number of training iterations
lr = 3e-4

# ===== Initialize model =====
vocab_size = encoding.n_vocab

def train_model():
    model = GPT(vocab_size=vocab_size, block_size=block_size, embed_dim=embed_dim, 
                num_heads=num_heads, FFN_depth=FFN_depth, encoder_layers=encoder_layers)
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    print(f"Using device: {device}")

    optimizer = optim.AdamW(model.parameters(), lr=lr)

    # ===== Training loop =====
    for epoch in range(epochs):
        # Random batch
        random_chunks = torch.randint(0, len(X) - block_size, (batch_size,))
        x_batch = torch.stack([X[i:i+block_size] for i in random_chunks]).to(torch.long).to(device)
        y_batch = torch.stack([Y[i:i+block_size] for i in random_chunks]).to(torch.long).to(device)

        # Forward pass
        logits = model(x_batch, y_batch)             # (B, T, vocab)
        loss = F.cross_entropy(logits.view(-1, vocab_size), y_batch.view(-1))

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 50 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch}/{epochs} - Loss: {loss.item():.4f}")

    torch.save(model.state_dict(), "gpt_model_trained.pt")

if __name__ == "__main__":
    train_model()
