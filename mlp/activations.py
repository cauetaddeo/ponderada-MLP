import numpy as np


def relu(z):
    return np.maximum(0.0, z)


def relu_derivative(z):
    return (z > 0.0).astype(z.dtype)


def tanh(z):
    return np.tanh(z)


def tanh_derivative(z):
    a = np.tanh(z)
    return 1.0 - a * a


def get_activation(name):
    activations = {
        "relu": (relu, relu_derivative),
        "tanh": (tanh, tanh_derivative),
    }
    try:
        return activations[name]
    except KeyError as exc:
        choices = ", ".join(sorted(activations))
        raise ValueError(f"Ativacao desconhecida: {name}. Use uma destas: {choices}") from exc
