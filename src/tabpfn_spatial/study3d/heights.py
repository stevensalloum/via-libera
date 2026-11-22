"""When gas density changes where the gas goes, and so whether people should crouch.

Britter & McQuaid (1988, "Workbook on the dispersion of dense gases", HSE CR 17/1988): a
continuous release behaves as a dense gas when

    ( g'0 q0 / (U^3 D) )^(1/3) >= 0.15,   g'0 = g |SG - 1|,   D = sqrt(q0 / U),

with q0 the volume release rate (m^3/s) and U the wind speed (m/s). Below 0.15 the release is
passive: it follows the air, whatever its density. For gases lighter than air we apply the
same magnitude (an approximation, flagged in the report: buoyant rise is normally judged
with the Briggs buoyancy flux, which gives the same g' q0 scaling).

The criterion is evaluated with the *pure* gas at the hole, so it flags almost every gas.
What a planner needs is **how far from the leak density still matters**: the dense (or
buoyant) phase ends when the density excess of the mixture falls below ~1 % of air (Britter &
McQuaid's passive limit), i.e. at volume fraction c_p = 0.01 / |SG - 1|. With near-field plume
dilution, centreline c = q0 / (pi U sigma^2) (ground source, reflected) and sigma = i x
(Taylor, short times; turbulence intensity i = 0.10, a typical indoor / near-ground value),
that distance is x_p = sqrt(q0 / (pi U c_p)) / i. Within x_p, height matters (heavy gas hugs
the floor, light gas rises); beyond it the gas follows the air.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from tabpfn_spatial.study3d.overview import INK, MUTED, _style

G = 9.81
GASES = {"hydrogen": 0.07, "methane": 0.55, "ammonia": 0.59, "CO": 0.97, "H2S": 1.19,
         "propane": 1.52, "CO2": 1.52, "butane": 2.0, "chlorine": 2.45}  # fmt: skip
CRITICAL = 0.15
INTENSITY = 0.10  # turbulence intensity sigma / U; ASSUMED typical (Red:Vapor measured 0.09)


def passive_distance(sg: float, q0: float, u: float, i: float = INTENSITY) -> float:
    """Distance (m) from the leak after which the gas is passive (density no longer matters)."""
    if abs(sg - 1) < 1e-9:
        return 0.0
    cp = min(0.01 / abs(sg - 1), 1.0)
    return float(np.sqrt(q0 / (np.pi * u * cp)) / i)


def criterion(sg: float, q0: float, u: float) -> float:
    """Britter-McQuaid dense-gas number for a continuous release."""
    d = np.sqrt(q0 / u)
    return float((G * abs(sg - 1) * q0 / (u**3 * d)) ** (1 / 3))


def fig_heights(out: Path) -> Path:
    """Distance from the leak within which density decides height, gases x (rate, speed)."""
    rates = [0.0001, 0.001, 0.01, 0.1]  # m^3/s: 0.1 L/s (small fitting leak) .. 100 L/s
    speeds = [0.1, 0.3, 1.0, 3.0]
    cols = [(q, u) for q in rates for u in speeds]
    vals = np.array([[passive_distance(sg, q, u) for q, u in cols] for sg in GASES.values()])
    fig, ax = plt.subplots(figsize=(13, 4.8))
    im = ax.imshow(np.log10(np.clip(vals, 0.05, None)), cmap="Blues", vmin=-1.3, vmax=2.3,
                   aspect="auto")  # fmt: skip
    for i in range(vals.shape[0]):
        for j in range(vals.shape[1]):
            v = vals[i, j]
            ax.text(j, i, f"{v:.1f}" if v < 10 else f"{v:.0f}", ha="center", va="center",
                    fontsize=7, color="white" if v > 20 else INK)  # fmt: skip
    ax.set_yticks(range(len(GASES)), [f"{g} (SG {sg})" for g, sg in GASES.items()], fontsize=8)
    ax.set_xticks(range(len(cols)), [f"{q * 1000:g} L/s\n{u:g} m/s" for q, u in cols], fontsize=7)
    ax.set_xlabel("leak rate (volume of pure gas per second) and air speed")
    cb = fig.colorbar(im, ax=ax, shrink=0.8)
    cb.set_label("log10 distance (m)", color=INK)
    title = "Within how many metres of the leak does gas density decide height?"
    ax.set_title(title + " (beyond it the gas follows the air)", fontsize=10, color=INK, loc="left")
    _style(ax)
    ax.grid(False)
    fig.text(0.01, 0.01, "Britter & McQuaid (1988) passive limit: density excess < 1 % of air; "
             "plume dilution with 10 % turbulence intensity. SG = density relative to air.",
             fontsize=7, color=MUTED)  # fmt: skip
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


if __name__ == "__main__":
    print(fig_heights(Path("docs/figures/3d-study/heights.png")))
