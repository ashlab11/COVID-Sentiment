import os
import pandas as pd
import nltk
import gensim
import torch

nltk.download('punkt')

print(os.getcwd())

model = gensim.models.KeyedVectors.load_word2vec_format("features_extraction/GoogleNews-vectors-negative300.bin", binary=True)
results = []

data = pd.read_csv("clean_COVIDSenti.csv")

for tweet, label in zip(data["tweet"], data["label"]):
    tokens = nltk.word_tokenize(tweet)
    
    word_embeddings = []
    
    for j in tokens:
        if j in model:
            word_embeddings.append(model[j])
    if len(word_embeddings) == 0:
        continue
    results.append((word_embeddings, label))

torch.save(results, "word2vec.pt")