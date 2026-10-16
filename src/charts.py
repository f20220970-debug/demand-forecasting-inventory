import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import config as C


def _save(fig, name):
    C.CHARTS.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(C.CHARTS / name, dpi=120); plt.close(fig)


def total_forecast(Y, weeks, bt, method):
    total = Y.sum(axis=0)
    fig, ax = plt.subplots(figsize=(11, 4.2))
    ax.plot(weeks, total, color="#5A6270", lw=1, label="actual (all 20 items)")
    ax.plot(bt["test_weeks"], bt["forecasts"][method].sum(axis=0), color="#C0392B", lw=2, label=f"forecast: {method}")
    ax.plot(bt["test_weeks"], bt["forecasts"]["naive"].sum(axis=0), color="#9DB4C0", lw=1.5, ls="--", label="forecast: naive")
    ax.axvspan(bt["test_weeks"][0], bt["test_weeks"][-1], color="#EEF3F8")
    ax.set_ylabel("Units per week"); ax.legend(); ax.set_title("Weekly demand and one-week-ahead forecasts (shaded = hold-out quarter)")
    _save(fig, "forecast_vs_actual.png")


def method_wape(overall):
    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.barh(overall["method"][::-1], 100 * overall["wape"][::-1], color="#2C7A7B")
    for i, v in enumerate(100 * overall["wape"][::-1]):
        ax.text(v + 0.5, i, f"{v:.1f}%", va="center")
    ax.set_xlabel("WAPE (%), lower is better"); ax.set_title("Forecast error by method (hold-out quarter)")
    _save(fig, "method_wape.png")


def abc_xyz_heatmap(counts, rev):
    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    ax.imshow(rev.values, cmap="Blues")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{counts.values[i, j]} items\n{100 * rev.values[i, j]:.1f}% rev", ha="center", va="center", fontsize=8)
    ax.set_xticks(range(3)); ax.set_xticklabels(["X stable", "Y variable", "Z erratic"])
    ax.set_yticks(range(3)); ax.set_yticklabels(["A top 80%", "B next 15%", "C last 5%"])
    ax.set_title("ABC-XYZ: value vs predictability")
    _save(fig, "abc_xyz.png")


def safety_curve(sv):
    fig, ax = plt.subplots(figsize=(7, 4))
    x = 100 * sv["service_level"]
    ax.plot(x, sv["naive"] / 1e3, marker="o", label="naive forecast")
    ax.plot(x, sv["selected"] / 1e3, marker="o", label="selected method")
    ax.set_xlabel("Service level (%)"); ax.set_ylabel("Safety stock value (thousand £)"); ax.legend()
    ax.set_title("Safety stock needed vs service level")
    _save(fig, "safety_stock.png")
