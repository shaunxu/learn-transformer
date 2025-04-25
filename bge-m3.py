from transformers import AutoTokenizer, AutoModel
import torch

path = "/Users/shaunxu/huggingface/bge-m3"
tokenizer = AutoTokenizer.from_pretrained(path)
model = AutoModel.from_pretrained(path)

print(model)