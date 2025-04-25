import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from datetime import datetime

token_to_id = {
    "what" : 0,
    "is" : 1,
    "pingcode" : 2,
    "awesome": 3,
    "<EOS>" : 4,
}
id_to_token = dict(map(reversed, token_to_id.items()))

inputs = torch.tensor([
    [
        token_to_id["what"],
        token_to_id["is"], 
        token_to_id["pingcode"], 
        token_to_id["<EOS>"],
        token_to_id["awesome"]
    ], 
    [
        token_to_id["pingcode"],
        token_to_id["is"], 
        token_to_id["what"], 
        token_to_id["<EOS>"], 
        token_to_id["awesome"]
    ]
])

labels = torch.tensor([
    [
        token_to_id["is"], 
        token_to_id["pingcode"], 
        token_to_id["<EOS>"], 
        token_to_id["awesome"], 
        token_to_id["<EOS>"]
    ],  
    [
        token_to_id["is"], 
        token_to_id["what"], 
        token_to_id["<EOS>"], 
        token_to_id["awesome"], 
        token_to_id["<EOS>"]
    ]
])

dataset = TensorDataset(inputs, labels) 
dataloader = DataLoader(dataset)

class PositionEncoding(nn.Module):

    def __init__(self, d_model=2, max_len=6, *args, **kwargs):
        super().__init__(*args, **kwargs)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(start=0, end=max_len, step=1).float().unsqueeze(1)
        embedding_index = torch.arange(start=0, end=d_model, step=2).float()
        div_term = 1/torch.tensor(10000.0)**(embedding_index / d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    
    def forward(self, word_embeddings):
        return word_embeddings + self.pe[:word_embeddings.size(0), :]

class Attention(nn.Module):

    def __init__(self, d_model=2, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.d_model=d_model
        self.W_q = nn.Linear(in_features=d_model, out_features=d_model, bias=False)
        self.W_k = nn.Linear(in_features=d_model, out_features=d_model, bias=False)
        self.W_v = nn.Linear(in_features=d_model, out_features=d_model, bias=False)
        self.row_dim = 0
        self.col_dim = 1
    
    def forward(self, encodings_for_q, encodings_for_k, encodings_for_v, mask=None):
        q = self.W_q(encodings_for_q)
        k = self.W_k(encodings_for_k)
        v = self.W_v(encodings_for_v)
        sims = torch.matmul(q, k.transpose(dim0=self.row_dim, dim1=self.col_dim))
        scaled_sims = sims / torch.tensor(k.size(self.col_dim)**0.5)
        if mask is not None:
            scaled_sims = scaled_sims.masked_fill(mask=mask, value=-1e9)
        attention_percents = F.softmax(scaled_sims, dim=self.col_dim)
        attention_scores = torch.matmul(attention_percents, v)
        return attention_scores

class DecoderOnlyTransformer(nn.Module):

    def __init__(self, num_tokens=4, d_model=2, max_len=6, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.we = nn.Embedding(num_embeddings=num_tokens, 
                               embedding_dim=d_model)
        self.pe = PositionEncoding(d_model=d_model, 
                                   max_len=max_len)
        self.self_attention = Attention(d_model=d_model)
        self.fc_layer = nn.Linear(in_features=d_model, out_features=num_tokens)
        self.loss = nn.CrossEntropyLoss()
    
    def forward(self, token_ids):
        word_embeddings = self.we(token_ids)
        position_encoded = self.pe(word_embeddings)
        mask = torch.tril(torch.ones((token_ids.size(dim=0), token_ids.size(dim=0))))
        mask = mask == 0
        self_attention_values = self.self_attention(position_encoded, 
                                                    position_encoded, 
                                                    position_encoded, 
                                                    mask=mask)
        residual_connection_values = position_encoded + self_attention_values
        fc_layer_output = self.fc_layer(residual_connection_values)
        return fc_layer_output

model = DecoderOnlyTransformer(num_tokens=len(token_to_id), d_model=2, max_len=6)

def generate(input_ids):
    input_length = input_ids.size(dim=0)
    print(input_ids)
    predictions = model(input_ids)
    predicted_id = torch.tensor([torch.argmax(predictions[-1,:])])

    predicted_ids = predicted_id
    max_length = 6
    for i in range(input_length, max_length):
        if predicted_id == token_to_id["<EOS>"]:
            break
        input_ids = torch.cat((input_ids, predicted_id))
        predictions = model(input_ids) 
        predicted_id = torch.tensor([torch.argmax(predictions[-1,:])])
        predicted_ids = torch.cat((predicted_ids, predicted_id))
    print("Predicted Tokens:") 
    for id in predicted_ids:
        print(id, id_to_token[id.item()])

NUM_EPOCHS = 30

def train():
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)

    for epoch in range(NUM_EPOCHS):
        total_loss = 0.0
        for i, (src, tgt) in enumerate(dataloader):
            src_data = src[0]
            tgt_data = tgt[0]
            optimizer.zero_grad()
            output = model(src_data)
            loss = criterion(output.view(-1, len(token_to_id)), tgt_data)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(dataloader)
        print(f'End of Epoch {epoch+1}, Average Loss: {avg_loss:.4f}')

def save():
    now = datetime.now().strftime("%Y%m%d%H%M%S")
    filename = f"checkpoints/{now}.pth"
    torch.save(model.state_dict(), filename)

def load(filename):
    model.load_state_dict(torch.load(f"checkpoints/{filename}"))

if __name__ == "__main__":
    # train()
    # save()
    # generate(torch.tensor([
    #     token_to_id["what"], 
    #     token_to_id["is"], 
    #     token_to_id["pingcode"], 
    #     token_to_id["<EOS>"]
    # ]))

    load("20250425162006.pth")
    generate(torch.tensor([
        token_to_id["what"], 
        token_to_id["is"], 
        token_to_id["pingcode"], 
        token_to_id["<EOS>"]
    ]))
    generate(torch.tensor([
        token_to_id["pingcode"], 
        token_to_id["is"], 
        token_to_id["what"], 
        token_to_id["<EOS>"]
    ]))
