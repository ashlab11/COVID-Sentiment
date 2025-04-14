import os
import pandas as pd
import nltk
import gensim
import torch

nltk.download('punkt_tab')

print(os.getcwd())

model = gensim.models.KeyedVectors.load_word2vec_format("features_extraction/GoogleNews-vectors-negative300.bin", binary=True)

# for l in ["A", "B", "C"]:
#     results = []
#     data = pd.read_csv("clean_COVIDSenti-" + l + ".csv")

#     for tweet, label in zip(data["tweet"], data["label"]):
#         tokens = nltk.word_tokenize(tweet)
        
#         word_embeddings = []
        
#         for j in tokens:
#             if j in model:
#                 word_embeddings.append(model[j])
#         results.append((word_embeddings, label))

#     torch.save(results, "features_extraction/word2vec_" + l + ".pt")

results = []

data = pd.read_csv("clean_COVIDSenti.csv")

print(data["tweet"][2161])


for tweet, label in zip(data["tweet"], data["label"]):
    tokens = nltk.word_tokenize(tweet)
    
    word_embeddings = []
    
    for j in tokens:
        if j in model:
            word_embeddings.append(model[j])
    if len(word_embeddings) == 0:
        continue
    results.append((word_embeddings, label))

torch.save(results, "features_extraction/word2vec.pt")




# loaded = torch.load("features_extraction/word2vec_A.pt", weights_only=False)