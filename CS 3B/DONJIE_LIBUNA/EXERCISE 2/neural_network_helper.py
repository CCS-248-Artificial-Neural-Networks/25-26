"""
neural_network_helper.py

Helper utilities for small feedforward neural network forward passes.
"""

from typing import Union
import numpy as np

# ---------- Setup / validation ----------
def setup_arrays(*arrays):
    converted = []
    for a in arrays:
        arr = np.array(a, dtype=float)
        if arr.ndim == 2 and (arr.shape[1] == 1):
            arr = arr.reshape(arr.shape[0],)
        converted.append(arr)
    return tuple(converted)

# ---------- Weighted sum + bias ----------
def weighted_sum(x, W, b):
    x, W, b = setup_arrays(x, W, b)
    W = np.array(W, dtype=float)
    if W.ndim != 2:
        raise ValueError("W must be 2D")
    if x.shape[0] != W.shape[0]:
        raise ValueError(f"Input length {x.shape[0]} does not match W rows {W.shape[0]}")
    if b.shape[0] != W.shape[1]:
        raise ValueError(f"Bias length {b.shape[0]} does not match W columns {W.shape[1]}")
    z = np.dot(x, W) + b
    return z

# ---------- Activation functions ----------
def relu(x):
    x = np.array(x, dtype=float)
    return np.maximum(0, x)

def sigmoid(x):
    x = np.array(x, dtype=float)
    return 1.0 / (1.0 + np.exp(-x))

def softmax(x):
    x = np.array(x, dtype=float)
    x_shift = x - np.max(x)
    ex = np.exp(x_shift)
    return ex / np.sum(ex)

def activation(z, kind='relu'):
    kind = kind.lower()
    if kind == 'relu':
        return relu(z)
    if kind == 'sigmoid':
        return sigmoid(z)
    if kind == 'softmax':
        return softmax(z)
    raise ValueError(f"Unknown activation kind: {kind}")

# ---------- Loss functions ----------
def mse_loss(pred, target):
    pred, target = setup_arrays(pred, target)
    return float(np.mean((np.array(pred) - np.array(target))**2))

def categorical_crossentropy(pred, target, eps=1e-12):
    pred, target = setup_arrays(pred, target)
    pred = np.clip(pred, eps, 1.0 - eps)
    return float(-np.sum(target * np.log(pred)))

def binary_crossentropy(pred, target, eps=1e-12):
    pred, target = setup_arrays(pred, target)
    pred = np.clip(pred, eps, 1.0 - eps)
    return float(np.mean(-(target * np.log(pred) + (1 - np.array(target)) * np.log(1 - pred))))

def loss(pred, target, kind='mse'):
    kind = kind.lower()
    if kind == 'mse':
        return mse_loss(pred, target)
    if kind in ('categorical_crossentropy', 'cross_entropy', 'categorical'):
        return categorical_crossentropy(pred, target)
    if kind in ('binary_crossentropy', 'binary', 'bce'):
        return binary_crossentropy(pred, target)
    raise ValueError(f"Unknown loss kind: {kind}")

# ---------- Convenience: full forward for single layer ----------
def forward_layer(x, W, b, activation_kind='relu'):
    z = weighted_sum(x, W, b)
    a = activation(z, activation_kind)
    return z, a
