"""Physics ladder L1-L3: steady 3D concentration of a continuous point release.

Every parameter is either measured for the case (wind profile, turbulence) or taken from
the literature, named where it is used.

- **L1, Gaussian plume** (Pasquill-Gifford form with Taylor's statistical theory for the
  spread): straight wind, no obstacles, ground (and ceiling) reflection by image sources.
  sigma^2(t) = sigma0^2 + 2 sigma_turb^2 T_L (t - T_L (1 - exp(-t / T_L))), t = s / U
  (Taylor 1921; exact for an exponential velocity autocorrelation).
- **L2, finite-volume transport around obstacles:** steady advection-diffusion on a voxel
  grid; buildings block the flow and the gas. The wind is the measured profile with the
  faces touching a building set to zero, *not* deflected. That wind is not mass-consistent
  (air "disappears" into walls), so transport uses the advective form u . grad(c): gas
  follows the wind and is lost where the wind runs into a wall instead of piling up there
  (a known artefact, kept on purpose: it shows what the next level buys). For a
  divergence-free wind (L3) the advective and conservative forms are identical.
- **L3, mass-consistent wind + L2 transport:** the wind is the smallest correction of the
  L2 wind that conserves mass around the buildings (Sherman 1978, MATHEW; the diagnostic
  wind model behind QUIC-URB): u = u0 + grad(lambda), div(u) = 0, lambda = 0 on open
  boundaries, no flow through walls. It deflects the wind around and over buildings but has
  no wakes or recirculation (those need CFD: L4 / OpenFOAM).

Turbulent diffusivity (L2, L3): K = sigma^2 T_L with the Lagrangian time scale of the
neutral surface layer T_L = 0.5 z / sigma_w (Hanna 1982), capped at ``l_max / sigma_w``
indoors, where eddies cannot be larger than the room.
"""

from dataclasses import dataclass, field

import numpy as np
from scipy import sparse
from scipy.sparse import linalg

from tabpfn_spatial.study3d.geometry import Grid

OPEN_TUNNEL = ("x0", "x1", "y0", "y1", "top")  # outdoor model in a big wind tunnel
CLOSED_DUCT = ("x0", "x1")  # a small tunnel: in at x0, out at x1
VENTILATED_ROOM = ("floor", "top")  # air supplied at the floor, extracted at the ceiling


@dataclass(frozen=True)
class Flow:
    """Wind and turbulence of one case, as tables over height z (m)."""

    direction: tuple[float, float]  # unit vector the wind blows towards
    z: np.ndarray
    speed: np.ndarray  # mean horizontal speed (m/s)
    sigma_v: np.ndarray  # crosswind velocity standard deviation (m/s)
    sigma_w: np.ndarray  # vertical velocity standard deviation (m/s)
    updraft: float = 0.0  # uniform vertical velocity (m/s), e.g. ventilation
    l_max: float = np.inf  # largest eddy length scale (m)
    open_faces: tuple[str, ...] = field(default=OPEN_TUNNEL)

    def at(self, name: str, z: np.ndarray | float) -> np.ndarray:
        """Table ``name`` interpolated at heights ``z`` (held constant outside the table)."""
        return np.interp(z, self.z, getattr(self, name))

    def t_l(self, z: np.ndarray | float) -> np.ndarray:
        """Lagrangian time scale (s): 0.5 z / sigma_w (Hanna 1982), eddies <= l_max."""
        sw = np.maximum(self.at("sigma_w", z), 1e-3)
        return np.minimum(0.5 * np.maximum(z, 0.02), self.l_max) / sw

    def diffusivity(self, z: np.ndarray, horizontal: bool) -> np.ndarray:
        """Eddy diffusivity K = sigma^2 T_L (m^2/s)."""
        s = self.at("sigma_v" if horizontal else "sigma_w", z)
        return s**2 * self.t_l(z)


def _taylor(sig: float, t_l: float, t: np.ndarray, sigma0: float) -> np.ndarray:
    tt = np.maximum(t, 0.0)
    return np.sqrt(sigma0**2 + 2 * sig**2 * t_l * (tt - t_l * (1 - np.exp(-tt / t_l))))


def plume(points: np.ndarray, src: np.ndarray, flow: Flow, sigma0: float = 0.03,
          ceiling: float | None = None) -> np.ndarray:  # fmt: skip
    """L1: concentration per unit release rate (s/m^3) at ``points`` (n, 3).

    Wind speed and turbulence are taken at the release height (the classic plume
    assumption); the centreline moves with ``flow.updraft``. Reflection at z = 0 and, if
    given, at the ceiling (one image each).
    """
    d = np.asarray(flow.direction, float)
    rel = np.asarray(points, float) - src
    s = rel[:, :2] @ d
    n = rel[:, :2] @ np.array([-d[1], d[0]])
    zs = float(src[2])
    u = max(float(flow.at("speed", zs)), 1e-3)
    t = s / u
    t_l = float(flow.t_l(zs))
    sy = _taylor(float(flow.at("sigma_v", zs)), t_l, t, sigma0)
    sz = _taylor(float(flow.at("sigma_w", zs)), t_l, t, sigma0)
    h = zs + flow.updraft * np.maximum(t, 0)
    z = np.asarray(points, float)[:, 2]
    vert = np.exp(-((z - h) ** 2) / (2 * sz**2)) + np.exp(-((z + h) ** 2) / (2 * sz**2))
    if ceiling is not None:
        vert += np.exp(-((z - (2 * ceiling - h)) ** 2) / (2 * sz**2))
    c = np.exp(-(n**2) / (2 * sy**2)) * vert / (2 * np.pi * u * sy * sz)
    return np.where(s > 0, c, 0.0)


@dataclass(frozen=True)
class Faces:
    """Face-normal velocities of a staggered grid: ux (nx+1, ny, nz), uy, uz."""

    ux: np.ndarray
    uy: np.ndarray
    uz: np.ndarray


def _blocked(solid: np.ndarray, axis: int) -> np.ndarray:
    """Faces along ``axis`` touching a solid cell (shape of that face array)."""
    pad = [(0, 0)] * 3
    pad[axis] = (1, 1)
    s = np.pad(solid, pad, constant_values=False)
    lo = [slice(None)] * 3
    hi = [slice(None)] * 3
    lo[axis], hi[axis] = slice(0, -1), slice(1, None)
    return s[tuple(lo)] | s[tuple(hi)]


def _closed_boundary(flow: Flow, shape: tuple[int, int, int]) -> list[np.ndarray]:
    """Per axis, a mask of boundary faces that are walls (not in ``flow.open_faces``)."""
    out = []
    for axis, (a, b) in enumerate([("x0", "x1"), ("y0", "y1"), ("floor", "top")]):
        fs = list(shape)
        fs[axis] += 1
        m = np.zeros(fs, bool)
        idx = [slice(None)] * 3
        idx[axis] = 0
        m[tuple(idx)] = a not in flow.open_faces
        idx[axis] = -1
        m[tuple(idx)] = b not in flow.open_faces
        out.append(m)
    return out


def base_wind(g: Grid, solid: np.ndarray, flow: Flow) -> Faces:
    """L2 wind: the measured profile on every face, zero on faces touching walls/buildings."""
    nx, ny, nz = g.shape
    zc = g.axis(2)
    sp = flow.at("speed", zc)
    d = flow.direction
    ux = np.broadcast_to(d[0] * sp, (nx + 1, ny, nz)).copy()
    uy = np.broadcast_to(d[1] * sp, (nx, ny + 1, nz)).copy()
    uz = np.full((nx, ny, nz + 1), flow.updraft)
    closed = _closed_boundary(flow, g.shape)
    for axis, u in enumerate((ux, uy, uz)):
        u[_blocked(solid, axis) | closed[axis]] = 0.0
    return Faces(ux, uy, uz)


def _face_pairs(g: Grid, solid: np.ndarray, flow: Flow):
    """Yield (axis, face array index, cell a or -1, cell b or -1) for every open face.

    Cell ids are positions in the fluid-cell numbering; -1 marks the domain outside an
    open boundary. Faces touching a solid cell or a wall are skipped.
    """
    ids = np.full(g.shape, -1)
    ids[~solid] = np.arange((~solid).sum())
    closed = _closed_boundary(flow, g.shape)
    for axis in range(3):
        pad = [(0, 0)] * 3
        pad[axis] = (1, 1)
        p = np.pad(ids, pad, constant_values=-1)
        lo = [slice(None)] * 3
        hi = [slice(None)] * 3
        lo[axis], hi[axis] = slice(0, -1), slice(1, None)
        a, b = p[tuple(lo)], p[tuple(hi)]
        ok = ~_blocked(solid, axis) & ~closed[axis] & ((a >= 0) | (b >= 0))
        yield axis, np.nonzero(ok), a[ok], b[ok]


def mass_consistent(g: Grid, solid: np.ndarray, flow: Flow, u0: Faces) -> Faces:
    """L3: u = u0 + grad(lambda) with div(u) = 0 in every fluid cell (Sherman 1978)."""
    n = int((~solid).sum())
    h = g.cell
    rows, cols, vals = [], [], []
    div = np.zeros(n)
    us = [u0.ux.copy(), u0.uy.copy(), u0.uz.copy()]
    pairs = list(_face_pairs(g, solid, flow))
    for axis, f, a, b in pairs:
        u = us[axis][f]
        np.add.at(div, a[a >= 0], u[a >= 0] / h)  # outflow of a
        np.add.at(div, b[b >= 0], -u[b >= 0] / h)  # inflow of b
        both = (a >= 0) & (b >= 0)
        for i, j in ((a, b), (b, a)):  # -Laplacian (SPD): +1/h^2 diagonal, -1/h^2 off
            ii = i >= 0
            rows += [i[ii]]
            cols += [i[ii]]
            vals += [np.full(ii.sum(), 1 / h**2)]
            rows += [i[both]]
            cols += [j[both]]
            vals += [np.full(both.sum(), -1 / h**2)]
    lap = sparse.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                            shape=(n, n))  # fmt: skip
    lam, info = linalg.cg(lap, -div, rtol=1e-8, maxiter=5000)
    if info:
        raise RuntimeError(f"mass-consistent wind did not converge ({info})")
    lam_out = np.append(lam, 0.0)  # index -1 -> outside, lambda = 0
    for axis, f, a, b in pairs:
        us[axis][f] = us[axis][f] - (lam_out[b] - lam_out[a]) / h
    return Faces(*us)


def divergence(g: Grid, solid: np.ndarray, flow: Flow, w: Faces) -> np.ndarray:
    """Net outflow per fluid cell (m^3/s per m^3): 0 for a mass-consistent wind."""
    div = np.zeros(int((~solid).sum()))
    for axis, f, a, b in _face_pairs(g, solid, flow):
        u = (w.ux, w.uy, w.uz)[axis][f]
        np.add.at(div, a[a >= 0], u[a >= 0] / g.cell)
        np.add.at(div, b[b >= 0], -u[b >= 0] / g.cell)
    return div


def transport(g: Grid, solid: np.ndarray, flow: Flow, wind: Faces, src: np.ndarray,
              rate: float = 1.0, k_field: np.ndarray | None = None) -> np.ndarray:  # fmt: skip
    """Steady concentration (nx, ny, nz) of a release ``rate`` (units/s) at ``src``."""
    mat, ids = transport_matrix(g, solid, flow, wind, k_field)
    rhs = np.zeros(mat.shape[0])
    rhs[ids[source_cell(g, solid, src)]] = rate  # units/s into the cell; unknowns units/m^3
    out = np.zeros(g.shape)
    out[~solid] = _solve(mat.tocsc(), rhs)
    return out


def transport_matrix(
    g: Grid, solid: np.ndarray, flow: Flow, wind: Faces, k_field: np.ndarray | None = None
) -> tuple[sparse.csr_matrix, np.ndarray]:
    """The steady transport operator A (A c = sources) and the fluid-cell numbering.

    ``k_field`` (nx, ny, nz), if given, replaces the flow's K(z): each face takes the mean
    of its two cells (L4: K = nu_t / Sc_t from the CFD).

    Finite volumes: first-order upwind advection, central diffusion with K(z) at the face,
    no flux through walls and buildings, outflow carries gas out of open boundaries, inflow
    brings clean air. Advective form: the conservative operator minus c div(u), so a
    non-mass-consistent wind (L2) neither creates nor piles up gas; zero change for L3.
    Solved with GMRES + incomplete-LU, or a direct solve if that fails.
    """
    n = int((~solid).sum())
    h, area = g.cell, g.cell**2
    rows, cols, vals = [], [], []

    def add(r: np.ndarray, c: np.ndarray, v: np.ndarray) -> None:
        rows.append(r)
        cols.append(c)
        vals.append(v)

    zc = g.axis(2)
    for axis, f, a, b in _face_pairs(g, solid, flow):
        u = (wind.ux, wind.uy, wind.uz)[axis][f]
        zf = g.lo[2] + h * f[2] if axis == 2 else zc[f[2]]
        if k_field is None:
            k = flow.diffusivity(zf, horizontal=axis < 2)
        else:
            kc = np.pad(k_field, [(1, 1) if ax == axis else (0, 0) for ax in range(3)], mode="edge")
            lo = [slice(None)] * 3
            hi = [slice(None)] * 3
            lo[axis], hi[axis] = slice(0, -1), slice(1, None)
            k = 0.5 * (kc[tuple(lo)] + kc[tuple(hi)])[f]
        dcoef = k * area / h
        up, dn = np.maximum(u, 0) * area, np.minimum(u, 0) * area
        both = (a >= 0) & (b >= 0)
        aa, bb = a[both], b[both]
        add(aa, aa, up[both] + dcoef[both])
        add(aa, bb, dn[both] - dcoef[both])
        add(bb, aa, -up[both] - dcoef[both])
        add(bb, bb, -dn[both] + dcoef[both])
        only_a = (a >= 0) & (b < 0)  # boundary on the high side of cell a: outflow if u > 0
        add(a[only_a], a[only_a], up[only_a])
        only_b = (b >= 0) & (a < 0)  # boundary on the low side of cell b: outflow if u < 0
        add(b[only_b], b[only_b], -dn[only_b])
    net_out = divergence(g, solid, flow, wind) * h**3  # m^3/s leaving each cell
    add(np.arange(n), np.arange(n), -net_out)
    mat = sparse.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                            shape=(n, n))  # fmt: skip
    ids = np.full(g.shape, -1)
    ids[~solid] = np.arange(n)
    return mat, ids


def adjoint(g: Grid, solid: np.ndarray, flow: Flow, wind: Faces,
            sensors: np.ndarray) -> np.ndarray:  # fmt: skip
    """Source-receptor sensitivity (Keats, Yee & Lien 2007): (n_sensors, nx, ny, nz).

    Entry [s, cell] is the steady reading at sensor s for a unit release in that cell
    (0 in solids). One factorisation of A^T and one back-substitution per sensor give the
    readings of every possible leak at once, instead of one forward solve per leak.
    """
    mat, ids = transport_matrix(g, solid, flow, wind)
    lu = linalg.splu(mat.T.tocsc())
    cells = g.index(sensors)
    out = np.zeros((len(sensors), *g.shape))
    for k, (i, j, m) in enumerate(cells):
        if ids[i, j, m] < 0:
            out[k] = np.nan  # sensor inside a building
            continue
        e = np.zeros(mat.shape[0])
        e[ids[i, j, m]] = 1.0
        out[k][~solid] = lu.solve(e)
    return out


def source_cell(g: Grid, solid: np.ndarray, src: np.ndarray) -> tuple[int, int, int]:
    """Cell of the release, moved up out of any solid.

    A hole on top of a tank model voxelises into the tank's top cell, so the release goes
    into the first air cell above it.
    """
    i, j, k = (int(v) for v in g.index(np.asarray(src)[None])[0])
    while solid[i, j, k]:
        k += 1
        if k >= g.shape[2]:
            raise ValueError(f"source {src} is inside a solid column")
    return i, j, k


def _solve(mat: sparse.csc_matrix, rhs: np.ndarray) -> np.ndarray:
    n = mat.shape[0]
    try:
        ilu = linalg.spilu(mat, drop_tol=1e-4, fill_factor=10, permc_spec="NATURAL")
        pre = linalg.LinearOperator((n, n), ilu.solve)
        x, info = linalg.gmres(mat, rhs, M=pre, rtol=1e-10, restart=60, maxiter=400)
        if info == 0:
            return x
    except RuntimeError:
        pass
    return linalg.spsolve(mat, rhs)


def sample(g: Grid, field_: np.ndarray, xyz: np.ndarray) -> np.ndarray:
    """Field value of the cell containing each point (n,)."""
    i = g.index(xyz)
    return field_[i[:, 0], i[:, 1], i[:, 2]]
