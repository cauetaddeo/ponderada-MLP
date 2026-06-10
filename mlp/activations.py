import numpy as np


def relu(z):
    # Mantem valores positivos e zera os negativos.
    return np.maximum(0.0, z)


def relu_derivative(z):
    # A derivada da ReLU decide por onde o gradiente continua passando.
    return (z > 0.0).astype(z.dtype)


def tanh(z):
    # Ativacao alternativa para comparar com ReLU.
    return np.tanh(z)


def tanh_derivative(z):
    # Derivada da tanh reaproveita a propria ativacao.
    a = np.tanh(z)
    return 1.0 - a * a


def get_activation(name):
    # Mapeia o nome escolhido para a funcao direta e sua derivada.
    activations = {
        "relu": (relu, relu_derivative),
        "tanh": (tanh, tanh_derivative),
    }
    try:
        return activations[name]
    except KeyError as exc:
        choices = ", ".join(sorted(activations))
        raise ValueError(f"Ativacao desconhecida: {name}. Use uma destas: {choices}") from exc
