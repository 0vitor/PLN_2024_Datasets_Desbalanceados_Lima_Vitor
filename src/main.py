import pandas as pd

# a)
df = pd.read_csv('spam.csv')

from transformers import GPT2Tokenizer, GPT2Model
import faiss
import torch

# Carregar modelo GPT-2
tokenizer = GPT2Tokenizer.from_pretrained("miguelvictor/python-gpt2-large")
model = GPT2Model.from_pretrained("miguelvictor/python-gpt2-large")

# Definir o token de padding
tokenizer.pad_token = tokenizer.eos_token

# Gerar embeddings
texts = df['text'].tolist()
inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)

# Passar os textos pelo modelo GPT-2 e obter as embeddings
with torch.no_grad():
    embeddings = model(**inputs).last_hidden_state.mean(dim=1).detach().numpy()

# Indexação com FAISS
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

print("Indexação FAISS concluída com sucesso!")
