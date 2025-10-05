import torch
import tiktoken
from model import GPT
from training import vocab_size, block_size, embed_dim, num_heads, FFN_depth, encoder_layers
import torch.nn.functional as F

encoding = tiktoken.encoding_for_model("gpt-4")

model = GPT(
    vocab_size=vocab_size,
    block_size=block_size,
    embed_dim=embed_dim,
    num_heads=num_heads,
    FFN_depth=FFN_depth,
    encoder_layers=encoder_layers
)

device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
model.load_state_dict(torch.load("gpt_model_trained.pt", map_location=device))
model = model.to(device)
model.eval()
print("Model loaded and ready.")

def generate(model, prompt, encoding, max_new_tokens=50, block_size=32, temperature=1.0, top_k=None):
    model.eval()
    device = next(model.parameters()).device

    # Encode the text into tokens
    tokens = torch.tensor(encoding.encode(prompt), dtype=torch.long, device=device).unsqueeze(0)

    for _ in range(max_new_tokens):
        tokens_cond = tokens[:, -block_size:]  # crop context
        logits = model(tokens_cond, tokens_cond)
        logits = logits[:, -1, :] / temperature  # (1, vocab_size)
        probs = F.softmax(logits, dim=-1)

        # Optional: top-k filtering
        if top_k is not None:
            values, indices = torch.topk(probs, top_k)
            probs = torch.zeros_like(probs).scatter_(1, indices, values)
            probs = probs / probs.sum(dim=-1, keepdim=True)

        next_token = torch.multinomial(probs, num_samples=1)
        tokens = torch.cat((tokens, next_token), dim=1)

    return encoding.decode(tokens[0].tolist())

prompt = input('How can I help you? ')
print(generate(model, prompt=prompt, encoding=encoding))

