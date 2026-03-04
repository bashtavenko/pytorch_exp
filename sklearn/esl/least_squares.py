"""
2D Linear Classification via Normal Equation (Least Squares). Thank you, Claude.

Encodes labels as y in {0,1} and fits a linear model via the closed-form:
    w = (X_b^T X_b)^-1 X_b^T y,  X_b = [X|1]  (bias appended)
Decision boundary: w^T x + b = 0.5
Reference: ESL Section 2.3.1  - Least Squares.

Usage:
    python neq.py
    python neq.py --n_points=15 --noise=0.35
    python neq.py --n_points=10 --noise=0.80
"""

import matplotlib
import numpy as np

matplotlib.use("qtagg")
import matplotlib.pyplot as plt
from absl import app, flags, logging

FLAGS = flags.FLAGS
flags.DEFINE_integer("n_points", 50, "Points per class.")
flags.DEFINE_float("noise", 2.0, "Gaussian noise std.")
flags.DEFINE_float("centre_gap", 2.0, "Half-distance between class centres.")
flags.DEFINE_string("output", "output/least_squares.png", "Output path.")
flags.DEFINE_boolean("show", False, "Show plot interactively.")


class NormalEquationClassifier:
    """Least-squares linear classifier via the normal equation."""

    def fit(self, X, y):
        X_b = np.hstack([X, np.ones((len(X), 1))])
        self.w_ = np.linalg.lstsq(X_b, y, rcond=None)[0]
        logging.info("w = %s  b = %.2f",
                     np.array2string(self.w_[:-1], precision=2), self.w_[-1])
        return self

    def decision_function(self, X):
        return np.hstack([X, np.ones((len(X), 1))]) @ self.w_

    def predict(self, X):
        return (self.decision_function(X) >= 0.5).astype(int)


def make_data(n, noise, gap, rng):
    # rng.normal(loc, scale, sze)
    # loc = mean, scale = std, sze = output shape
    X = np.vstack([rng.normal([-gap, 0], noise, (n, 2)),
                   rng.normal([gap, 0], noise, (n, 2))])
    y = np.array([0] * n + [1] * n)
    return X, y


def standardise(X):
    """Zero mean is not strictly required if features have the same scale."""
    mu, sigma = X.mean(0), X.std(0)
    return (X - mu) / sigma, mu, sigma


def plot(X, y, clf, mu, sigma, output_path, show):
    pad = 1.2
    xx, yy = np.meshgrid(
        np.linspace(X[:, 0].min() - pad, X[:, 0].max() + pad, 400),
        np.linspace(X[:, 1].min() - pad, X[:, 1].max() + pad, 400),
    )
    Z = clf.decision_function((np.c_[xx.ravel(), yy.ravel()] - mu) / sigma).reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.contourf(xx, yy, Z, levels=64, cmap="RdBu_r", alpha=0.25)
    ax.contour(xx, yy, Z, levels=[0.5], colors="k", linewidths=1.5, linestyles="--")
    ax.scatter(*X[y == 0].T, c="#2196F3", edgecolors="white", s=90, zorder=5, label="Class 0")
    ax.scatter(*X[y == 1].T, c="#F44336", edgecolors="white", s=90, zorder=5, label="Class 1")

    acc = np.mean(clf.predict((X - mu) / sigma) == y)
    ax.set_title("Normal Equation")
    ax.set_xlabel("$x_1$");
    ax.set_ylabel("$x_2$")
    ax.legend(loc="lower right")
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    logging.info("Saved to %s", output_path)
    if show:
        plt.show()
    plt.close(fig)


def main(argv):
    del argv
    rng = np.random.default_rng(seed=42);
    X, y = make_data(FLAGS.n_points, FLAGS.noise, FLAGS.centre_gap, rng)
    logging.info("X = %s  counts = %s", X.shape, np.bincount(y))

    X_s, mu, sigma = standardise(X)
    clf = NormalEquationClassifier().fit(X_s, y)
    logging.info("Train acc: %.1f%%", 100 * np.mean(clf.predict(X_s) == y))

    plot(X, y, clf, mu, sigma, FLAGS.output, FLAGS.show)


if __name__ == "__main__":
    app.run(main)
