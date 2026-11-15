"""Optional two-member accuracy experiment using the unchanged 30k preset.

Each member retains all four operators and the corrected, optimized pipeline.
The total feature budget and extraction work increase; this is not a free gain.
"""
import numpy as np
from multirocket.presets import make_multirocket


def series_view(X, view):
    X = np.asarray(X, dtype=np.float64)
    if view in ('identity', 'seed'):
        return X
    if view == 'normalized':
        scale = X.std(axis=1, keepdims=True)
        return (X - X.mean(axis=1, keepdims=True)) / np.where(scale > 1e-12, scale, 1.)
    if view == 'coarse':
        # Pair averaging suppresses high-frequency detail. Preserve minimum length.
        if X.shape[1] < 20:
            return X
        length = X.shape[1] // 2
        return X[:, :2 * length].reshape(len(X), length, 2).mean(axis=2)
    raise ValueError('Unknown view: ' + str(view))


def ridge_scores(model, X):
    """Folded ridge scores, extracting at most one prediction batch at a time."""
    X = np.asarray(X, dtype=np.float64)
    classes = model.classifier[-1].classes_
    columns = 1 if len(classes) <= 2 else len(classes)
    result = np.empty((len(X), columns), dtype=np.float64)
    for first in range(0, len(X), model.prediction_batch_size):
        chunk = X[first:first + model.prediction_batch_size]
        features = model._transform(chunk, np.diff(chunk, axis=1))
        np.nan_to_num(features, copy=False)
        result[first:first + len(chunk)] = features @ model.prediction_coef_.T + model.prediction_intercept_
    return result


def labels_from_scores(scores, classes):
    if len(classes) == 1:
        return np.full(len(scores), classes[0])
    indices = (scores[:, 0] > 0).astype(int) if scores.shape[1] == 1 else scores.argmax(axis=1)
    return classes[indices]


class MultiRocketAccuracyEnsemble:
    """Equal-score ensemble of the recommended model and one additional view."""
    def __init__(self, view='seed', random_state=20261111):
        if view not in ('seed', 'normalized', 'coarse'):
            raise ValueError('view must be seed, normalized, or coarse')
        self.view = view
        self.random_state = random_state

    def fit(self, X, y):
        self.members_ = []
        for index, view in enumerate(('identity', self.view)):
            seed = (int(self.random_state) + index * 104729) % (2 ** 32)
            model = make_multirocket(random_state=seed)
            model.fit(series_view(X, view), y, predict_on_train=False)
            self.members_.append(model)
        self.classes_ = self.members_[0].classifier[-1].classes_
        return self

    def decision_function(self, X):
        if not hasattr(self, 'members_'):
            raise ValueError('Call fit before prediction')
        return (ridge_scores(self.members_[0], X) +
                ridge_scores(self.members_[1], series_view(X, self.view))) / 2

    def predict(self, X):
        return labels_from_scores(self.decision_function(X), self.classes_)
