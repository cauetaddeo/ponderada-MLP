import numpy as np


class SGD:
    def __init__(self, learning_rate=0.01, momentum=0.0):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.velocity = {}

    def step(self, params, grads):
        for key, value in params.items():
            grad = grads[key]
            if self.momentum:
                velocity = self.velocity.get(key, np.zeros_like(value))
                velocity = self.momentum * velocity - self.learning_rate * grad
                params[key] += velocity
                self.velocity[key] = velocity
            else:
                params[key] -= self.learning_rate * grad
