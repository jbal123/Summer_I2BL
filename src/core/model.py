
# --- path bootstrap (auto-added during repo reorg): make sibling modules importable ---
import os as _os, sys as _sys
_SRC_DIR = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _sub in ("core", "features", "day1_da_only", "single_analyte", "multi_analyte", "models"):
    _p = _os.path.join(_SRC_DIR, _sub)
    if _os.path.isdir(_p) and _p not in _sys.path:
        _sys.path.insert(0, _p)
# --- end path bootstrap ---

import numpy as np


class ANN:
    def __init__(self, layer_sizes: list[int]):
        self.layer_sizes = layer_sizes
        self.weights = []
        self.biases  = []
        self._init_params()

    def _init_params(self):
        for i in range(len(self.layer_sizes) - 1):
            fan_in  = self.layer_sizes[i]
            fan_out = self.layer_sizes[i + 1]
            w = np.random.randn(fan_in, fan_out) * np.sqrt(2.0 / fan_in)
            b = np.zeros((1, fan_out))
            self.weights.append(w)
            self.biases.append(b)

    def relu(self, z):
        return np.maximum(0, z)

    def relu_derivative(self, z):
        return (z > 0).astype(float)

    def mse_loss(self, y_pred, y_true):
        return float(np.mean((y_pred - y_true) ** 2))

    def mse_loss_derivative(self, y_pred, y_true):
        return 2 * (y_pred - y_true) / y_true.size

    def asymmetric_mse_loss(self, y_pred, y_true, alpha=2.0):
        residuals = y_pred - y_true
        weights   = np.where(residuals < 0, alpha, 1.0)
        return float(np.mean(weights * residuals ** 2))

    def asymmetric_mse_loss_derivative(self, y_pred, y_true, alpha=2.0):
        residuals = y_pred - y_true
        weights   = np.where(residuals < 0, alpha, 1.0)
        return 2 * weights * residuals / y_true.size

    def forward(self, X):
        cache = []
        a = X
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            z = a @ w + b
            a = self.relu(z) if i < len(self.weights) - 1 else z
            cache.append((z, a))
        return a, cache

    def backward(self, X, y_true, cache, loss_deriv=None):
        m = X.shape[0]
        grad_weights = [None] * len(self.weights)
        grad_biases  = [None] * len(self.biases)

        y_pred = cache[-1][1]
        delta  = (loss_deriv or self.mse_loss_derivative)(y_pred, y_true)

        for i in reversed(range(len(self.weights))):
            z, a   = cache[i]
            a_prev = cache[i - 1][1] if i > 0 else X
            if i < len(self.weights) - 1:
                delta = delta * self.relu_derivative(z)
            grad_weights[i] = a_prev.T @ delta / m
            grad_biases[i]  = delta.mean(axis=0, keepdims=True)
            delta            = delta @ self.weights[i].T

        return grad_weights, grad_biases

    def update_params(self, grad_weights, grad_biases, lr):
        for i in range(len(self.weights)):
            self.weights[i] -= lr * grad_weights[i]
            self.biases[i]  -= lr * grad_biases[i]

    def train(self, X, y, epochs=1000, lr=0.001, verbose=True,
              X_val=None, y_val=None, val_every=100, patience=0, max_epochs=None,
              asymmetric=False, alpha=2.0):
        """
        Train for at least `epochs` epochs.  If `patience` > 0 and val data is
        provided, continue past `epochs` until val loss fails to improve for
        `patience` consecutive val checks (each separated by `val_every` epochs).
        `max_epochs` is a hard ceiling on total epochs regardless of patience.
        `asymmetric=True` penalises underprediction by `alpha`x.
        """
        _loss_fn    = (lambda yp, yt: self.asymmetric_mse_loss(yp, yt, alpha)) if asymmetric else self.mse_loss
        _loss_deriv = (lambda yp, yt: self.asymmetric_mse_loss_derivative(yp, yt, alpha)) if asymmetric else self.mse_loss_derivative

        history = {'epochs': [], 'train_loss': [], 'val_loss': []}

        can_extend = patience > 0 and X_val is not None
        best_val   = float('inf')
        no_improve = 0

        epoch = 0
        while True:
            epoch += 1
            y_pred, cache = self.forward(X)
            loss          = _loss_fn(y_pred, y)
            gw, gb        = self.backward(X, y, cache, loss_deriv=_loss_deriv)
            self.update_params(gw, gb, lr)

            if epoch % val_every == 0:
                history['epochs'].append(epoch)
                history['train_loss'].append(loss)
                if X_val is not None:
                    val_pred, _ = self.forward(X_val)
                    v_loss      = self.mse_loss(val_pred, y_val)
                    history['val_loss'].append(v_loss)
                    if verbose:
                        ext = ' [ext]' if epoch > epochs else ''
                        print(f"  Epoch {epoch:>7}  Train: {loss:.6f}  Val: {v_loss:.6f}{ext}")
                    if epoch >= epochs and can_extend:
                        if v_loss < best_val:
                            best_val   = v_loss
                            no_improve = 0
                        else:
                            no_improve += 1
                elif verbose:
                    print(f"  Epoch {epoch:>7}  Loss: {loss:.6f}")

            # Stopping conditions
            if max_epochs is not None and epoch >= max_epochs:
                break
            if epoch >= epochs:
                if not can_extend or no_improve >= patience:
                    break

        return history

    def predict(self, X):
        out, _ = self.forward(X)
        return out


def make_model(n_features: int) -> ANN:
    h1 = max(16, n_features // 2)
    h2 = max(8,  n_features // 4)
    return ANN(layer_sizes=[n_features, h1, h2, 3])
