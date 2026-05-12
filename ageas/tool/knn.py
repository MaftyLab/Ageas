#!/usr/bin/env python3
"""Lightweight tensor-only KNN and KMeans helpers.

Self-contained PyTorch implementations used as cluster/classifier fallbacks
where a full dependency on scikit-learn is undesirable.

Adapted from
https://gist.github.com/wzjoriv/7e89afc7f30761022d7747a501260fe3.
"""
import torch


class KNN:
    """Tensor-only k-nearest-neighbor classifier with majority voting.

    Attributes:
        k: Number of nearest neighbors.
        ord: Norm order used for distance computation.
        train_pts: Reference training points tensor.
        train_label: Labels corresponding to ``train_pts``.
        unique_labels: Unique label values found in ``train_label``.
    """

    def __init__(self, X: torch.Tensor, Y: torch.Tensor, k: int = 3, ord: int = 2) -> None:
        """Initialize KNN with training data.

        Args:
            X: Training feature matrix of shape ``(n, d)``.
            Y: Training label vector of length ``n``.
            k: Number of nearest neighbors to consider.
            ord: Order of the vector norm for distance computation.
        """
        self.k = k
        self.ord = ord
        self.train_pts = X
        self.train_label = Y
        self.unique_labels = self.train_label.unique()

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Predict class labels for query points via majority voting.

        Args:
            x: Query points of shape ``(m, d)``.

        Returns:
            Predicted label tensor of length ``m``.
        """
        dist = self.distance_matrix(x, self.train_pts, self.ord)
        knn = dist.topk(self.k, largest=False)
        votes = self.train_label[knn.indices]

        winner = torch.zeros(votes.size(0), dtype=votes.dtype, device=votes.device)
        count = torch.zeros(votes.size(0), dtype=votes.dtype, device=votes.device) - 1

        for lab in self.unique_labels:
            vote_count = (votes == lab).sum(1)
            who = vote_count >= count
            winner[who] = lab
            count[who] = vote_count[who]

        return winner

    def distance_matrix(
        self,
        x: torch.Tensor,
        y: torch.Tensor = None,
        ord: int = 2,
    ) -> torch.Tensor:
        """Compute a pairwise vector distance matrix.

        Args:
            x: Query points of shape ``(n, d)``.
            y: Reference points of shape ``(m, d)``. ``None`` reuses ``x``.
            ord: Order of the vector norm forwarded to
                :func:`torch.linalg.vector_norm`.

        Returns:
            Distance matrix of shape ``(n, m)``.
        """
        y = x if y is None else y

        n = x.size(0)
        m = y.size(0)
        d = x.size(1)

        x = x.unsqueeze(1).expand(n, m, d)
        y = y.unsqueeze(0).expand(n, m, d)

        return torch.linalg.vector_norm(x - y, ord, dim=2)


class KMeans(KNN):
    """Tensor-only KMeans clustering driven by the parent's distance matrix.

    Attributes:
        k: Number of clusters.
        n_iters: Number of centroid update iterations.
        ord: Norm order used for distance computation.
        train_pts: Current cluster centroid positions.
        train_label: Integer cluster indices.
    """

    def __init__(
        self,
        X: torch.Tensor,
        k: int = 2,
        n_iters: int = 10,
        ord: int = 2,
    ) -> None:
        """Initialize KMeans and run centroid updates.

        Args:
            X: Input data matrix of shape ``(n, d)``.
            k: Number of clusters.
            n_iters: Number of centroid update iterations.
            ord: Norm order for distance computation.
        """
        self.k = k
        self.n_iters = n_iters
        self.ord = ord

        self.train_pts = X[torch.randperm(len(X))[: self.k]]
        self.train_label = torch.LongTensor(range(self.k))

        for _ in range(self.n_iters):
            labels = self.predict(X)
            for lab in range(self.k):
                select = labels == lab
                self.train_pts[lab] = torch.mean(X[select], dim=0)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Assign cluster labels to query points.

        Args:
            x: Query points of shape ``(m, d)``.

        Returns:
            Cluster label tensor of length ``m``.
        """
        return self.predict(x)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Assign each query point to its nearest centroid.

        Args:
            x: Query points of shape ``(m, d)``.

        Returns:
            Cluster label tensor of length ``m``.
        """
        dist = self.distance_matrix(x, self.train_pts, self.ord)
        labels = torch.argmin(dist, dim=1)
        return self.train_label[labels]


if __name__ == '__main__':
    a = torch.Tensor([[1, 1], [0.88, 0.90], [-1, -1], [-1, -0.88]])
    b = torch.LongTensor([3, 3, 5, 5])
    c = torch.Tensor([[-0.5, -0.5], [0.88, 0.88]])

    knn = KNN(a, b)
    print(knn(c))
