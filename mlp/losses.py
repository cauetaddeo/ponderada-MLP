import numpy as np


def softmax(logits):
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp_scores = np.exp(shifted)
    return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)


def cross_entropy(probs, y_true):
    n_samples = y_true.shape[0]
    clipped = np.clip(probs, 1e-12, 1.0)
    return -np.mean(np.log(clipped[np.arange(n_samples), y_true]))


def accuracy(probs, y_true):
    return float(np.mean(np.argmax(probs, axis=1) == y_true))


def one_hot(y, n_classes):
    encoded = np.zeros((y.shape[0], n_classes), dtype=np.float64)
    encoded[np.arange(y.shape[0]), y] = 1.0
    return encoded
