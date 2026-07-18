"""A small, classic PyTorch MLP classifier with a scikit-learn-like API.

Module summary
--------------
The BERT engine's head. Rather than a plain logistic regression on the
sentence embeddings, we train a proper (if compact) **multi-layer
perceptron** in PyTorch — the textbook "embedding → dense → ReLU →
dropout → softmax" classifier. It is wrapped behind a
``fit`` / ``predict_proba`` interface so the engine code reads exactly like
it would with any scikit-learn estimator, and so it can be swapped for one
in tests.

Architecture (deliberately classic, nothing exotic):

    input(dim)  →  LayerNorm  →  Linear(dim, hidden)  →  ReLU  →  Dropout
                →  Linear(hidden, n_classes)  →  softmax (via CrossEntropy)

Trained with Adam + cross-entropy, a fixed seed for reproducibility, and
early-ish stopping via a fixed epoch budget that is plenty for a few
hundred embedded sentences.

Usage example
-------------
>>> import numpy as np
>>> from intent_engine.mlp import TorchMLPClassifier
>>> x = np.random.RandomState(0).randn(20, 8)
>>> y = np.array(["a", "b"] * 10)
>>> clf = TorchMLPClassifier(epochs=5).fit(x, y)   # doctest: +SKIP
>>> clf.predict_proba(x).shape                      # doctest: +SKIP
(20, 2)

Author
------
Project maintainers.
"""

from __future__ import annotations

import numpy as np


class TorchMLPClassifier:
    """A compact PyTorch MLP behind a scikit-learn-style estimator API.

    Parameters
    ----------
    hidden : int, optional
        Width of the single hidden layer, by default 256.
    dropout : float, optional
        Dropout probability after the hidden activation, by default 0.3.
    epochs : int, optional
        Number of full-batch training epochs, by default 200.
    lr : float, optional
        Adam learning rate, by default 1e-3.
    weight_decay : float, optional
        Adam L2 regularisation, by default 1e-4 (guards against overfitting
        the few hundred training vectors).
    seed : int, optional
        RNG seed for reproducible weights/training, by default 0.

    Attributes
    ----------
    classes_ : np.ndarray
        Sorted unique labels, populated by :meth:`fit` (mirrors sklearn).
    """

    def __init__(
        self,
        hidden: int = 256,
        dropout: float = 0.3,
        epochs: int = 200,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        seed: int = 0,
    ) -> None:
        """Store hyper-parameters; the network is built lazily in ``fit``."""
        # Hyper-parameters, kept as attributes so a fitted model can be
        # inspected/serialised and so tests can shrink ``epochs``.
        self.hidden = hidden
        self.dropout = dropout
        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.seed = seed
        # Populated during fit.
        self.classes_: np.ndarray = np.asarray([])
        self._model = None
        # Maps each label to its integer class index for the loss function.
        self._label_to_idx: dict[str, int] = {}

    def _build(self, input_dim: int, n_classes: int):
        """Construct the MLP network for a given input/output shape.

        Parameters
        ----------
        input_dim : int
            Embedding dimensionality (e.g. 384 for MiniLM).
        n_classes : int
            Number of intents.

        Returns
        -------
        torch.nn.Module
            The (untrained) network.
        """
        # Import torch lazily so importing this module never forces the heavy
        # dependency on a machine that only runs the non-neural engines.
        import torch
        from torch import nn

        # Seed every torch RNG so the same data yields the same weights and
        # training trajectory — reproducibility is part of the contract.
        torch.manual_seed(self.seed)
        # The classic head: normalise the embedding, one hidden ReLU layer
        # with dropout for regularisation, then a linear projection to logits.
        return nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, self.hidden),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden, n_classes),
        )

    def fit(self, x: np.ndarray, y: np.ndarray) -> TorchMLPClassifier:
        """Train the MLP on embedded features ``x`` and string labels ``y``.

        Parameters
        ----------
        x : np.ndarray
            ``(n_samples, dim)`` float feature matrix (sentence embeddings).
        y : np.ndarray
            ``(n_samples,)`` array of string intent ids.

        Returns
        -------
        TorchMLPClassifier
            ``self``, fitted.
        """
        import torch
        from torch import nn

        # Establish a stable class ordering (sorted) so ``predict_proba``
        # columns are deterministic and align with ``classes_``.
        self.classes_ = np.array(sorted(set(y.tolist())))
        self._label_to_idx = {label: i for i, label in enumerate(self.classes_)}

        # Convert features and integer-encoded labels to tensors.
        features = torch.tensor(np.asarray(x), dtype=torch.float32)
        targets = torch.tensor(
            [self._label_to_idx[label] for label in y], dtype=torch.long
        )

        # Build the network and the standard classification training setup.
        self._model = self._build(features.shape[1], len(self.classes_))
        optimizer = torch.optim.Adam(
            self._model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        loss_fn = nn.CrossEntropyLoss()

        # Full-batch gradient descent: the dataset is tiny (a few hundred
        # vectors), so batching adds noise for no speed-up. Train mode keeps
        # dropout active for regularisation.
        self._model.train()
        for _ in range(self.epochs):
            optimizer.zero_grad()
            logits = self._model(features)
            loss = loss_fn(logits, targets)
            loss.backward()
            optimizer.step()
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """Return class probabilities for each row of ``x``.

        Parameters
        ----------
        x : np.ndarray
            ``(n_samples, dim)`` feature matrix.

        Returns
        -------
        np.ndarray
            ``(n_samples, n_classes)`` softmax probabilities aligned with
            :attr:`classes_`.

        Raises
        ------
        RuntimeError
            If called before :meth:`fit`.
        """
        import torch

        if self._model is None:
            raise RuntimeError("TorchMLPClassifier.predict_proba called before fit().")
        # Eval mode disables dropout for a deterministic forward pass; no grad
        # needed at inference.
        self._model.eval()
        with torch.no_grad():
            features = torch.tensor(np.asarray(x), dtype=torch.float32)
            probabilities = torch.softmax(self._model(features), dim=1)
        return probabilities.numpy()
