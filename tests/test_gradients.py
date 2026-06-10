import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    # Garante imports locais quando o teste roda fora da raiz do projeto.
    sys.path.insert(0, str(PROJECT_ROOT))

from mlp import MLP


def test_gradient_check_small_network():
    # Rede pequena e dados sinteticos deixam o gradient check rapido e deterministico.
    rng = np.random.default_rng(7)
    x = rng.normal(size=(5, 4))
    y = np.array([0, 1, 2, 1, 0])
    model = MLP([4, 6, 5, 3], seed=3)
    probs, cache = model.forward(x)
    assert probs.shape == (5, 3)

    grads = model.backward(cache, y)
    epsilon = 1e-5
    # Limita os parametros checados para manter o teste leve no CI/local.
    max_checks = 20
    checked = 0

    for param_name, param in model.params.items():
        for index in np.ndindex(param.shape):
            original = param[index]
            param[index] = original + epsilon
            loss_plus = model.compute_loss(x, y)
            param[index] = original - epsilon
            loss_minus = model.compute_loss(x, y)
            param[index] = original

            # Gradiente numerico por diferencas finitas deve bater com o backprop.
            numerical = (loss_plus - loss_minus) / (2 * epsilon)
            analytical = grads[param_name][index]
            assert np.isclose(numerical, analytical, atol=1e-5)

            checked += 1
            if checked >= max_checks:
                return


if __name__ == "__main__":
    test_gradient_check_small_network()
    print("gradient check ok")
