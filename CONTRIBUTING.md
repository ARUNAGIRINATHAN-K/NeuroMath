# Contributing to NeuroMath

Thank you for your interest in contributing to **NeuroMath**! Our goal is to make the mathematical foundations of artificial intelligence and neural networks accessible, rigorous, and completely transparent from first principles.

Whether you are fixing a typo, adding mathematical derivations, optimizing NumPy vectorization, or contributing a new laboratory module, your help is welcome.

---

## Code of Conduct

By participating in this project, you agree to abide by our [Code of Conduct](CODE_OF_CONDUCT.md). Please be respectful and collaborative.

---

## Development Setup

1. **Fork the repository** on GitHub.
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/<your-username>/NeuroMath.git
   cd NeuroMath
   ```
3. **Create a virtual environment**:
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```
4. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   pip install pytest flake8
   ```

---

## Architectural Principles

When writing or reviewing code for NeuroMath, adhere to these non-negotiable standards:

1. **First-Principles First**: Core forward activations, backpropagation passes, and parameter updates must be implemented in pure NumPy. Do not rely on high-level autograd frameworks (PyTorch, TensorFlow, JAX) for core algorithms unless it is strictly for benchmarking against scratch implementations.
2. **Vectorized Linear Algebra**: Avoid nested Python loops over samples wherever matrix dot products (`np.dot` or `@`) can be utilized.
3. **Explicit Derivatives**: Show or cite the analytical gradient derivations in the code comments or companion doc.
4. **Cross-Platform Path Safety**: Never hardcode absolute system paths (e.g., `/home/...` or `C:\...`). Use `os.path` or `pathlib` relative to `__file__`.

---

## Submitting Pull Requests

1. Create a feature branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```
2. Write unit tests under `tests/` to verify your changes.
3. Run the test suite:
   ```bash
   pytest tests/
   ```
4. Commit your changes with clear, descriptive commit messages:
   ```bash
   git commit -m "feat(optimizers): implement vectorized Adam optimizer in pure NumPy"
   ```
5. Push to your fork and submit a Pull Request targeting `main`.

---

## Need Help?

Feel free to open an issue or start a conversation in [GitHub Discussions](https://github.com/ARUNAGIRINATHAN-K/NeuroMath/discussions).

