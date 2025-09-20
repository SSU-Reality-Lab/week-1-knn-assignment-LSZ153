import numpy as np
from typing import Tuple

class KNearestNeighbor:
    """
    k-Nearest Neighbor classifier for CIFAR-10 (or any vector data).

    This implementation purposely avoids scikit-learn and similar libraries.
    It offers three distance computation modes to compare speed/accuracy:
      - compute_distances_two_loops
      - compute_distances_one_loop
      - compute_distances_no_loops  (fully vectorized)
    """

    def __init__(self) -> None:
        self.X_train: np.ndarray | None = None
        self.y_train: np.ndarray | None = None

    # ------------------------------------------------------------
    # API
    # ------------------------------------------------------------
    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        'Train' the classifier by simply memorizing the training data.

        Parameters
        ----------
        X : array, shape (N_train, D)
            Training features (flattened images).
        y : array, shape (N_train,)
            Integer labels in [0, C).
        """
        if X.ndim != 2:
            raise ValueError(f"X must be 2D (N, D). Got shape {X.shape}")
        if y.ndim != 1:
            raise ValueError(f"y must be 1D (N,). Got shape {y.shape}")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of samples.")
        # Cast to float32 to avoid uint8 overflow during squared-distance calcs
        self.X_train = X.astype(np.float32, copy=False)
        self.y_train = y.astype(np.int64, copy=False)

    def predict(self, X: np.ndarray, k: int = 1, num_loops: int = 0) -> np.ndarray:
        """
        Predict labels for test data using the stored training set.

        Parameters
        ----------
        X : array, shape (N_test, D)
        k : int
            Number of neighbors to vote over.
        num_loops : int
            Selects distance routine:
              2 -> compute_distances_two_loops
              1 -> compute_distances_one_loop
              0 -> compute_distances_no_loops (default)
        """
        if self.X_train is None or self.y_train is None:
            raise RuntimeError("Call train(X, y) before predict().")
        if k <= 0:
            raise ValueError("k must be a positive integer.")

        X_test = X.astype(np.float32, copy=False)

        if num_loops == 2:
            dists = self.compute_distances_two_loops(X_test)
        elif num_loops == 1:
            dists = self.compute_distances_one_loop(X_test)
        elif num_loops == 0:
            dists = self.compute_distances_no_loops(X_test)
        else:
            raise ValueError("num_loops must be one of {0, 1, 2}.")

        return self.predict_labels(dists, k=k)

    # ------------------------------------------------------------
    # Distance computations
    # ------------------------------------------------------------
    def compute_distances_two_loops(self, X: np.ndarray) -> np.ndarray:
        """
        Compute the L2 distance between each test point in X and each training
        point in self.X_train using a double for-loop.

        Returns
        -------
        dists : array, shape (N_test, N_train)
        """
        if self.X_train is None:
            raise RuntimeError("No training data. Call train() first.")
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.empty((num_test, num_train), dtype=np.float32)

        for i in range(num_test):
            for j in range(num_train):
                # L2 distance: ||x - y||_2
                diff = X[i] - self.X_train[j]
                dists[i, j] = np.sqrt(np.dot(diff, diff))
        return dists

    def compute_distances_one_loop(self, X: np.ndarray) -> np.ndarray:
        """
        Compute the L2 distance using a single loop over the test data.
        """
        if self.X_train is None:
            raise RuntimeError("No training data. Call train() first.")
        num_test = X.shape[0]
        dists = np.empty((num_test, self.X_train.shape[0]), dtype=np.float32)

        # Expand training data once for speed
        Xtr_sq = np.sum(self.X_train ** 2, axis=1)  # (N_train,)

        for i in range(num_test):
            # Using (a-b)^2 = a^2 + b^2 - 2ab
            xi = X[i]  # (D,)
            xi_sq = np.sum(xi ** 2)                   # scalar
            cross = self.X_train @ xi                 # (N_train,)
            # numerical guard: negative zeros -> clip at 0
            dists[i] = np.sqrt(np.maximum(xi_sq + Xtr_sq - 2.0 * cross, 0.0))
        return dists

    def compute_distances_no_loops(self, X: np.ndarray) -> np.ndarray:
        """
        Fully vectorized L2 distance for all test/train pairs.

        Uses the identity: ||x - y||^2 = ||x||^2 + ||y||^2 - 2 x·y

        Returns
        -------
        dists : array, shape (N_test, N_train)
        """
        if self.X_train is None:
            raise RuntimeError("No training data. Call train() first.")

        # (N_test, 1)
        X_sq = np.sum(X ** 2, axis=1, keepdims=True)
        # (1, N_train)
        Xtr_sq = np.sum(self.X_train ** 2, axis=1, keepdims=True).T
        # (N_test, N_train)
        cross = X @ self.X_train.T
        d2 = X_sq + Xtr_sq - 2.0 * cross
        # Numerical stability: distances must be >= 0
        np.maximum(d2, 0.0, out=d2)
        dists = np.sqrt(d2, dtype=np.float32)
        return dists

    # ------------------------------------------------------------
    # Voting
    # ------------------------------------------------------------
    def predict_labels(self, dists: np.ndarray, k: int = 1) -> np.ndarray:
        """
        Given a matrix of distances, predict a label for each test example.
        Tie-breaking is handled by choosing the smallest label index.

        Parameters
        ----------
        dists : array, shape (N_test, N_train)
        k : int
            Number of nearest neighbors

        Returns
        -------
        y_pred : array, shape (N_test,)
        """
        if self.y_train is None:
            raise RuntimeError("No training labels. Call train() first.")

        num_test = dists.shape[0]
        y_pred = np.empty(num_test, dtype=np.int64)

        max_label = int(np.max(self.y_train)) if self.y_train.size else 0
        for i in range(num_test):
            # Indices of k smallest distances
            nn_idx = np.argpartition(dists[i], kth=k-1)[:k]
            nn_labels = self.y_train[nn_idx]
            # Majority vote with deterministic tie-break (smallest label wins)
            counts = np.bincount(nn_labels, minlength=max_label + 1)
            y_pred[i] = np.argmax(counts)
        return y_pred


# ---------------------- Utility helpers (optional) ----------------------
def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute classification accuracy in [0, 1].
    """
    if y_true.shape != y_pred.shape:
        raise ValueError("Shapes of y_true and y_pred must match.")
    return float(np.mean(y_true == y_pred))


def train_val_split(X: np.ndarray, y: np.ndarray, val_ratio: float = 0.2, seed: int | None = 42
                    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Simple shuffled train/val split.
    """
    if not (0.0 < val_ratio < 1.0):
        raise ValueError("val_ratio must be in (0, 1).")
    N = X.shape[0]
    rng = np.random.default_rng(seed)
    perm = rng.permutation(N)
    split = int(N * (1.0 - val_ratio))
    idx_train, idx_val = perm[:split], perm[split:]
    return X[idx_train], y[idx_train], X[idx_val], y[idx_val]
