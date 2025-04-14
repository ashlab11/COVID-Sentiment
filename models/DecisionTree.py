import torch
from math import log2

class DecisionTree:
    def __init__(self, max_depth=5, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None
        
    def fit(self, X, y, depth=0):
        if depth >= self.max_depth or X.shape[0] < self.min_samples_split:
            self.tree = torch.mode(y).values.item()
            return self
        
        best_split = self.find_best_split(X, y)
        
        if best_split is None:
            self.tree = torch.mode(y).values.item()
            return self
            
        feature, threshold, left_indices, right_indices = best_split
        
        
        
        self.tree = {
            'feature': feature,
            'threshold': threshold,
            'left': DecisionTree(self.max_depth, self.min_samples_split).fit(X[left_indices], y[left_indices], depth + 1),
            'right': DecisionTree(self.max_depth, self.min_samples_split).fit(X[right_indices], y[right_indices], depth + 1),
        }
        
        return self
    
    def predict(self, X):
        if isinstance(self.tree, dict):
            print(X)
            
            if X[self.tree['feature']] <= self.tree['threshold']:
                return self.tree['left'].predict(X)
            else:
                return self.tree['right'].predict(X)
        else:
            return torch.tensor(self.tree)
        
        
    def calculate_entropy(self, y):
        # Caluclate the entropy of y
        # -(p1 * log(p1) + p2 * log(p2) + p3 * log(p))
        
        n_neg_ones = len([i for i in y if i == -1])
        n_ones = len([i for i in y if i == 1])
        n_zeros = len([i for i in y if i == 0])
        
        # Avoid log(0)
        eps = 1e-10
        n_samples = len(y)
        
        prob_neg_ones = n_neg_ones * 1.0 / n_samples + eps
        prob_ones = n_ones * 1.0 / n_samples + eps
        prob_zeros = n_zeros * 1.0 / n_samples + eps
        
        return -(
            prob_neg_ones * log2(prob_neg_ones) + prob_ones * log2(prob_ones) + prob_zeros * log2(prob_zeros)
        )
        
    def find_best_split(self, X, y):
        n_samples = len(X)
        n_features = len(X[0])
        
        current_entropy = self.calculate_entropy(y)
        current_gain = 0
        best_split = None
        
        for feature in range(n_features):
            features = X[:, feature]
            
            # Find the best split
            split_values = torch.unique(features)
            
            for s in split_values:
                left_indices = (features <= s).nonzero().reshape(-1)
                right_indices = (features > s).nonzero().reshape(-1)
                
                # If s can nont divides the data, skip
                if len(left_indices) == 0 or len(right_indices) == 0:
                    continue
                
                left_entropy = self.calculate_entropy(y[left_indices])
                right_entropy = self.calculate_entropy(y[right_indices])
                
                average_entropy = (len(left_indices) / n_samples) * left_entropy + (len(right_indices) / n_samples) * right_entropy
                
                gain = current_entropy - average_entropy
                
                if gain > current_gain:
                    current_gain = gain
                    best_split = (feature, s.item(), left_indices, right_indices)
        return best_split