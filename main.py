import numpy as np
import tiktoken
import torch.optim as optim
import torch
from model import positional_encoding, GPT

with open("data.txt") as file:
    data = file.read()

encoding = tiktoken.encoding_for_model("gpt-4")
tokens = encoding.encode(data)
print("Token count:", len(tokens))

