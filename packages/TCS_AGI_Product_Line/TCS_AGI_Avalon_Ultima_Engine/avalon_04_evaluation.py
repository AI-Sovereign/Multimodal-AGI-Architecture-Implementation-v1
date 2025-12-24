import numpy as np
from sklearn.neighbors import NearestNeighbors
import torch

class RecursiveHindsight:
    """Episodic Relational Memory Librarian."""
    def __init__(self, dim=256, capacity=5000):
        self.bank = np.zeros((capacity, dim))
        self.ptr = 0
        self.knn = NearestNeighbors(n_neighbors=1, metric='cosine')

    def store(self, latent):
        self.bank[self.ptr] = latent.detach().cpu().numpy().flatten()
        self.ptr = (self.ptr + 1) % len(self.bank)

    def retrieve(self, query):
        if self.ptr < 10: return torch.zeros(1, 256)
        self.knn.fit(self.bank[:max(self.ptr, 100)])
        _, idx = self.knn.kneighbors(query.detach().cpu().numpy())
        return torch.tensor(self.bank[idx[0][0]]).unsqueeze(0).float()
