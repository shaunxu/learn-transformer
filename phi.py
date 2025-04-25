from transformers import AutoTokenizer, AutoModel
import torch

path = "/Users/shaunxu/huggingface/Phi-4-mini-instruct"
tokenizer = AutoTokenizer.from_pretrained(path)
model = AutoModel.from_pretrained(path)

print(model)

# prompt = "The capital of China is"
# inputs = tokenizer(prompt, return_tensors="pt")
# input_ids = inputs.input_ids
# attention_mask = inputs.attention_mask
# print("input_ids", input_ids)
# print("attention_mask", attention_mask)

# seq_len = input_ids.size(1)
# position_ids = torch.arange(seq_len).unsqueeze(0)
# position_embeddings = model.embed_positions(position_ids)
# print(position_embeddings)

# input_tensor = torch.tensor(input_ids)

# embeddings = model.embed_tokens(input_tensor)
# print("embeddings", embeddings)
# print("embeddings (shape)", embeddings.shape)

# self_attn = model.layers[0].self_attn(embeddings, attention_mask=attention_mask)
# print(self_attn)