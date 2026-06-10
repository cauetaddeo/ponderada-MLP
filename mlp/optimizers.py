import numpy as np


class SGD:
    def __init__(self, learning_rate=0.01, momentum=0.0):
        self.learning_rate = learning_rate
        self.momentum = momentum
        # Guarda velocidades anteriores quando momentum esta ativo.
        self.velocity = {}

    def step(self, params, grads):
        # Atualiza cada parametro usando o gradiente calculado no backward.
        for key, value in params.items():
            grad = grads[key]
            if self.momentum:
                # Momentum suaviza a trajetoria somando parte do passo anterior.
                velocity = self.velocity.get(key, np.zeros_like(value))
                velocity = self.momentum * velocity - self.learning_rate * grad
                params[key] += velocity
                self.velocity[key] = velocity
            else:
                params[key] -= self.learning_rate * grad
