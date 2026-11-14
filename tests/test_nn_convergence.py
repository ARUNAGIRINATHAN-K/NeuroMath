import numpy as np
import pytest

class TestNeuralNetworkConvergence:
    def test_xor_mlp_convergence(self):
        """Verify that a 2-layer MLP converges on the XOR problem."""
        sigmoid = lambda z: 1 / (1 + np.exp(-z))
        sigmoid_derivative = lambda a: a * (1 - a)

        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
        y = np.array([[0], [1], [1], [0]], dtype=float)

        np.random.seed(42)
        W1 = np.random.randn(2, 4)
        b1 = np.zeros((1, 4))
        W2 = np.random.randn(4, 1)
        b2 = np.zeros((1, 1))

        lr = 0.5
        epochs = 3000

        for _ in range(epochs):
            # Forward
            Z1 = np.dot(X, W1) + b1
            A1 = sigmoid(Z1)
            Z2 = np.dot(A1, W2) + b2
            A2 = sigmoid(Z2)

            # Backward
            loss = np.mean((y - A2) ** 2)
            dZ2 = 2 * (A2 - y) * sigmoid_derivative(A2)
            dW2 = np.dot(A1.T, dZ2)
            db2 = np.sum(dZ2, axis=0, keepdims=True)

            dZ1 = np.dot(dZ2, W2.T) * sigmoid_derivative(A1)
            dW1 = np.dot(X.T, dZ1)
            db1 = np.sum(dZ1, axis=0, keepdims=True)

            # Update
            W2 -= lr * dW2
            b2 -= lr * db2
            W1 -= lr * dW1
            b1 -= lr * db1

        # Check convergence
        final_loss = np.mean((y - A2) ** 2)
        assert final_loss < 0.05, f"Loss did not converge: {final_loss}"
        predictions = (A2 > 0.5).astype(int)
        np.testing.assert_array_equal(predictions, y.astype(int))

    def test_inverted_dropout_expectation(self):
        """Verify inverted dropout maintains expected activation magnitude during training."""
        np.random.seed(42)
        activations = np.ones((1000, 100))
        p_drop = 0.4
        keep_prob = 1.0 - p_drop

        mask = (np.random.rand(*activations.shape) >= p_drop).astype(float)
        dropped = (activations * mask) / keep_prob

        # Expected value should remain approximately 1.0
        np.testing.assert_allclose(np.mean(dropped), 1.0, atol=0.05)

