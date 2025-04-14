import torch
import numpy as np
def average_embeddings(path):
    data = torch.load(path, weights_only=False)
    results = []
    for d in data:
        average = np.mean(d[0], axis=0)
        results.append((average, d[1]))
    return results
        
    
if __name__ == "__main__":
    test = average_embeddings("features_extraction/word2vec.pt")
    print(test[2161])