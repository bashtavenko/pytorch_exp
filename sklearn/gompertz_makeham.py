"""
Gompertz–Makeham Law of Mortality Visualization. Thank you, Claude.
It is a parametric statistical model, not a machine learning model.

Usage:
    python gompertz_makeham.py
    python gompertz_makeham.py --age_max=110 --age_step=2 --no_show
"""

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from absl import app
from absl import flags
from absl import logging

FLAGS = flags.FLAGS
flags.DEFINE_integer("age_max", 100, "Maximum age for simulation and plotting.")
flags.DEFINE_integer("age_step", 5, "Step size (years) between observed age samples.")
flags.DEFINE_float("true_lambda", 0.001, "True Makeham constant λ for data simulation.")
flags.DEFINE_float("true_A", 0.00004, "True Gompertz coefficient A for data simulation.")
flags.DEFINE_float("true_B", 0.085, "True Gompertz rate B for data simulation.")
flags.DEFINE_float("noise_frac", 0.08, "Fraction of hazard used as Gaussian noise std.")
flags.DEFINE_integer("random_seed", 42, "Random seed for reproducibility.")
flags.DEFINE_string("output", "gompertz_makeham.png", "Path to save the output figure.")
flags.DEFINE_boolean("show", False, "Display the plot interactively.")


# ── Custom scikit-learn Estimator ─────────────────────────────────────────────

class GompertzMakehamModel(BaseEstimator, TransformerMixin):
    """Fits and predicts the Gompertz–Makeham hazard rate: h(x) = λ + A·exp(B·x)."""

    def __init__(self, lambda_=0.001, A=0.00005, B=0.09):
        self.lambda_ = lambda_
        self.A = A
        self.B = B

    @staticmethod
    def _hazard(x, lambda_, A, B):
        return lambda_ + A * np.exp(B * x)

    def fit(self, X, y):
        popt, _ = curve_fit(
            self._hazard, X.ravel(), y,
            p0=[self.lambda_, self.A, self.B],
            bounds=([0, 0, 0], [1, 1, 1]),
            maxfev=10000
        )
        self.lambda_, self.A, self.B = popt
        return self

    def predict(self, X):
        return self._hazard(X.ravel(), self.lambda_, self.A, self.B)

    def survival(self, X):
        """Probability of surviving past age x: S(x) = exp(-integral_0^x h(t) dt)"""
        ages = X.ravel()
        S = np.zeros_like(ages, dtype=float)
        for i, age in enumerate(ages):
            t = np.linspace(0, age, 1000)
            S[i] = np.exp(-np.trapezoid(self._hazard(t, self.lambda_, self.A, self.B), t))
        return S

    def density(self, X):
        """Distribution of age at death f(x) = h(x) * S(x)"""
        return self.predict(X) * self.survival(X)


def main(argv):
    del argv  # unused

    np.random.seed(FLAGS.random_seed)
    logging.info("Random seed set to %d", FLAGS.random_seed)

    # ── Simulate mortality data ───────────────────────────────────────────────
    ages_obs = np.arange(0, FLAGS.age_max, FLAGS.age_step, dtype=float)
    true_h = FLAGS.true_lambda + FLAGS.true_A * np.exp(FLAGS.true_B * ages_obs)
    observed = np.clip(
        true_h + np.random.normal(0, true_h * FLAGS.noise_frac),
        1e-6, None
    )
    logging.info("Simulated %d age samples from 0 to %d", len(ages_obs), FLAGS.age_max)
    logging.info("True params  →  λ=%.6f  A=%.6f  B=%.6f",
                 FLAGS.true_lambda, FLAGS.true_A, FLAGS.true_B)

    # ── scikit-learn Pipeline ─────────────────────────────────────────────────
    pipeline = Pipeline([
        ("identity", FunctionTransformer()),
        ("gm_model", GompertzMakehamModel()),
    ])
    pipeline.fit(ages_obs.reshape(-1, 1), observed)
    model = pipeline.named_steps["gm_model"]

    logging.info("Fitted params →  λ=%.6f  A=%.6f  B=%.6f",
                 model.lambda_, model.A, model.B)

    # Quick prediction examples
    for age in [30, 65, 80]:
        h = model.predict(np.array([[age]]))[0]
        s = model.survival(np.array([[age]]))[0]
        logging.info("Age %3d  →  h(x)=%.6f   S(x)=%.4f", age, h, s)

    # ── Build smooth curves ───────────────────────────────────────────────────
    ages_s = np.linspace(0, FLAGS.age_max, 500).reshape(-1, 1)
    h_pred = model.predict(ages_s)
    S_pred = model.survival(ages_s)
    f_pred = model.density(ages_s)

    # ── Visualize ─────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        "Gompertz–Makeham Law of Mortality\n"
        f"h(x) = λ + A·exp(Bx)  |  "
        f"λ={model.lambda_:.5f},  A={model.A:.5f},  B={model.B:.4f}",
        fontsize=13, fontweight="bold"
    )

    # (a) Hazard Rate
    ax = axes[0, 0]
    ax.scatter(ages_obs, observed, color="steelblue", s=60, zorder=5, label="Observed (simulated)")
    ax.plot(ages_s, h_pred, color="crimson", lw=2.5, label="Fitted h(x)")
    ax.axhline(model.lambda_, color="gray", ls="--", lw=1.2,
               label=f"Makeham λ={model.lambda_:.4f}")
    ax.set(title="Hazard Rate  h(x)", xlabel="Age", ylabel="h(x)")
    ax.legend();
    ax.grid(alpha=0.3)

    # (b) Log Hazard Rate
    ax = axes[0, 1]
    ax.scatter(ages_obs, np.log(observed), color="steelblue", s=60, zorder=5,
               label="Observed log-hazard")
    ax.plot(ages_s, np.log(h_pred), color="crimson", lw=2.5, label="Fitted log h(x)")
    ax.set(title="Log Hazard Rate  log h(x)", xlabel="Age", ylabel="log h(x)")
    ax.legend();
    ax.grid(alpha=0.3)

    # (c) Survival Function
    ax = axes[1, 0]
    ax.plot(ages_s, S_pred, color="seagreen", lw=2.5)
    ax.fill_between(ages_s.ravel(), S_pred, alpha=0.15, color="seagreen")
    ax.set(title="Survival Function  S(x)", xlabel="Age", ylabel="S(x)", ylim=(0, 1.02))
    ax.grid(alpha=0.3)

    # (d) Lifetime Density
    ax = axes[1, 1]
    ax.plot(ages_s, f_pred, color="darkorange", lw=2.5)
    ax.fill_between(ages_s.ravel(), f_pred, alpha=0.2, color="darkorange")
    ax.set(title="Lifetime Density  f(x) = h(x)·S(x)", xlabel="Age", ylabel="f(x)")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(FLAGS.output, dpi=150, bbox_inches="tight")
    logging.info("Figure saved to %s", FLAGS.output)

    if FLAGS.show:
        plt.show()


if __name__ == "__main__":
    app.run(main)
