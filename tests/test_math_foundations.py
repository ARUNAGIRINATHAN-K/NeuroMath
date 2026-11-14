import numpy as np
import pytest

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(z):
    s = sigmoid(z)
    return s * (1 - s)

def relu(z):
    return np.maximum(0, z)

def relu_derivative(z):
    return (z > 0).astype(float)

def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=-1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=-1, keepdims=True)

def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def mse_derivative(y_true, y_pred):
    return 2 * (y_pred - y_true) / y_true.size

class TestMathFoundations:
    def test_sigmoid_values_bounded(self):
        x = np.array([-100.0, -1.0, 0.0, 1.0, 100.0])
        s = sigmoid(x)
        assert np.all(s >= 0.0) and np.all(s <= 1.0)
        assert np.isclose(sigmoid(0.0), 0.5)

    def test_sigmoid_derivative_analytical_vs_numerical(self):
        """Verify analytical derivative matches finite difference approximation."""
        x = np.array([-2.5, -0.5, 0.0, 0.75, 2.0])
        eps = 1e-6
        analytical = sigmoid_derivative(x)
        numerical = (sigmoid(x + eps) - sigmoid(x - eps)) / (2 * eps)
        np.testing.assert_allclose(analytical, numerical, rtol=1e-5, atol=1e-5)

    def test_relu_derivative(self):
        x = np.array([-3.0, -0.1, 0.5, 4.0])
        expected = np.array([0.0, 0.0, 1.0, 1.0])
        np.testing.assert_array_equal(relu_derivative(x), expected)

    def test_softmax_probability_distribution(self):
        logits = np.array([[2.0, 1.0, 0.1], [1000.0, 1000.0, 1000.0]])
        probs = softmax(logits)
        # Sum of probabilities across classes must be 1.0
        np.testing.assert_allclose(np.sum(probs, axis=-1), np.array([1.0, 1.0]), rtol=1e-6)
        assert np.all(probs >= 0.0) and np.all(probs <= 1.0)

    def test_mse_and_derivative_zero_at_target(self):
        y_true = np.array([[1.0], [0.0]])
        y_pred = np.array([[1.0], [0.0]])
        assert mean_squared_error(y_true, y_pred) == 0.0
        np.testing.assert_array_equal(mse_derivative(y_true, y_pred), np.zeros_like(y_true))

