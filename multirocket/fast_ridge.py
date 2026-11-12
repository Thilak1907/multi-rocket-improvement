"""Dense wide-data ridge with sklearn's default leave-one-out MSE objective.

Build the Gram matrix from float64 column blocks instead of making two full
float64 feature copies. Compute coefficients only for the winning alpha, not
for every candidate. No approximation, randomized solver, or float32 Gram.
Tall inputs use sklearn to avoid allocating an unnecessarily large Gram matrix.
"""
import numpy as np
from scipy.linalg import eigh
from scipy.linalg.blas import dsyrk
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import RidgeClassifierCV
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted


class FastRidgeClassifierCV(ClassifierMixin, BaseEstimator):
    """Unweighted, dense ridge classification with an unpenalized intercept.

    This deliberately implements only the default MultiRocket use case:
    ``cv=None``, ``scoring=None``, ``fit_intercept=True``, no class/sample weights.
    ``block_size`` bounds the float64 feature workspace, not the O(n²) Gram.
    """

    def __init__(self, alphas=None, block_size=1024):
        self.alphas = alphas
        self.block_size = block_size

    def fit(self, X, y):
        X, y = check_X_y(X, y, dtype=[np.float32, np.float64], ensure_min_samples=2)
        alphas = np.asarray(np.logspace(-3, 3, 10) if self.alphas is None else self.alphas, dtype=np.float64)
        if alphas.ndim != 1 or not len(alphas) or not np.all(np.isfinite(alphas)) or np.any(alphas <= 0):
            raise ValueError("alphas must be a nonempty vector of finite positive values")
        if not isinstance(self.block_size, (int, np.integer)) or self.block_size < 1:
            raise ValueError("block_size must be a positive integer")
        n, p = X.shape
        self.n_features_in_ = p
        if n > p:
            reference = RidgeClassifierCV(alphas=alphas).fit(X, y)
            for name in ("classes_", "coef_", "intercept_", "alpha_", "best_score_"):
                setattr(self, name, getattr(reference, name))
            self.solver_ = "sklearn_tall"
            return self

        encoder = LabelBinarizer(pos_label=1, neg_label=-1)
        Y = encoder.fit_transform(y).astype(np.float64)
        self.classes_ = encoder.classes_
        y_mean = Y.mean(axis=0)
        Y -= y_mean
        x_mean = X.mean(axis=0, dtype=np.float64)
        gram = np.zeros((n, n), dtype=np.float64, order="F")
        for start in range(0, p, self.block_size):
            stop = min(p, start + self.block_size)
            block = np.array(X[:, start:stop], dtype=np.float64, order="F")
            block -= x_mean[start:stop]
            gram = dsyrk(1.0, block, beta=1.0, c=gram, lower=1, overwrite_c=1)
        del block
        eigenvalues, Q = eigh(gram, lower=True, overwrite_a=True, check_finite=False)
        del gram
        QT_y = Q.T @ Y
        QT_one = Q.sum(axis=0)
        inv = 1.0 / (eigenvalues[:, None] + alphas[None, :])
        # The intercept is unpenalized: subtract G^-1 1 / n from its diagonal.
        # Omitting this term selects a different alpha even when labels agree.
        diagonal = np.empty((n, len(alphas)), dtype=np.float64)
        for start in range(0, n, 256):
            q = Q[start:start + 256]
            diagonal[start:start + 256] = (q * q) @ inv - q @ (inv * QT_one[:, None]) / n
        scores = np.empty(len(alphas))
        for i in range(len(alphas)):
            dual = Q @ (inv[:, i, None] * QT_y)
            errors = dual / diagonal[:, i, None]
            scores[i] = -np.mean(errors * errors)
        # np.argmax keeps the first exact tie, as sklearn does.
        best = int(np.argmax(scores))
        # The blocked Gram calculation can change the ordering of alphas whose
        # LOO scores differ only at roundoff scale. For those rare cases, use
        # sklearn's own calculation to settle the choice exactly as the
        # reference estimator would. Its cost is paid only for a near tie.
        if len(scores) > 1:
            two_best = np.partition(scores, -2)[-2:]
            if two_best[1] - two_best[0] <= 1e-7 * max(1.0, abs(scores[best])):
                reference = RidgeClassifierCV(alphas=alphas).fit(X, y)
                for name in ("classes_", "coef_", "intercept_", "alpha_", "best_score_"):
                    setattr(self, name, getattr(reference, name))
                self.loo_scores_ = scores
                self.solver_ = "sklearn_near_tie"
                return self
        self.alpha_ = alphas[best]
        self.best_score_ = scores[best]
        self.loo_scores_ = scores
        dual = Q @ (inv[:, best, None] * QT_y)
        del Q, inv, diagonal
        coef = np.empty((p, Y.shape[1]), dtype=np.float64)
        for start in range(0, p, self.block_size):
            stop = min(p, start + self.block_size)
            block = np.array(X[:, start:stop], dtype=np.float64)
            block -= x_mean[start:stop]
            coef[start:stop] = block.T @ dual
        self.intercept_ = (y_mean - x_mean @ coef).astype(X.dtype)
        self.coef_ = coef.T.astype(X.dtype)
        self.solver_ = "blocked_gram"
        return self

    def decision_function(self, X):
        check_is_fitted(self, "coef_")
        X = check_array(X, dtype=[np.float32, np.float64])
        if X.shape[1] != self.n_features_in_:
            raise ValueError("X has a different number of features from training")
        scores = X @ np.atleast_2d(self.coef_).T + self.intercept_
        return scores.ravel() if scores.shape[1] == 1 else scores

    def predict(self, X):
        scores = self.decision_function(X)
        if len(self.classes_) == 1:
            return np.full(len(X), self.classes_[0])
        indices = (scores > 0).astype(int) if scores.ndim == 1 else scores.argmax(axis=1)
        return self.classes_[indices]
