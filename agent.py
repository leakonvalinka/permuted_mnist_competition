"""
Softmax Linear Classification Agent for Permuted MNIST
Uses PyTorch for a single linear layer with softmax (logistic regression).
"""
import numpy as np
import torch
import torch.nn as nn


class Agent:
    """Linear softmax classifier using PyTorch SGD."""

    def __init__(self, input_dim: int = 784, output_dim: int = 10, seed: int = None,
                 learning_rate: float = 0.05, epochs: int = 5, batch_size: int = 256):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.seed = seed
        if seed is not None:
            torch.manual_seed(seed)
        self.device = torch.device("cpu")

        self.model = nn.Linear(self.input_dim, self.output_dim).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)

    def reset(self):
        """Reset model weights for a new task."""
        if self.seed is not None:
            torch.manual_seed(self.seed)
        self._init_model()

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train the linear model on the provided data."""
        X = torch.from_numpy(X_train / 255.).to(self.device)
        y = torch.from_numpy(y_train.ravel().astype(np.int64)).to(self.device)

        n_samples = X.shape[0]
        self.model.train()

        for epoch in range(self.epochs):
            perm = torch.randperm(n_samples)
            for i in range(0, n_samples, self.batch_size):
                idx = perm[i:i + self.batch_size]
                X_batch = X[idx]
                y_batch = y[idx]

                logits = self.model(X_batch)
                loss = self.criterion(logits, y_batch)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

    @torch.no_grad()
    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Predict class labels for test data."""
        X = self._preprocess(X_test)
        self.model.eval()
        logits = self.model(X)
        return logits.argmax(dim=1).cpu().numpy()
