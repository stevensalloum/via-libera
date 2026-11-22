"""L4 wind: OpenFOAM RANS (simpleFoam, standard k-epsilon) around the Red:Vapor site.

Domain = the DNW test section around the turntable: x in [-2.4, 2.4] m (the wind blows
towards -x, so the inlet is at +2.4), y in [-1.5, 1.5] (3.0 m wide), z in [0, 2.25].
- Inlet: neutral atmospheric boundary layer (Richards & Hoxey 1993): U = u*/kappa ln((z+z0)/z0),
  k = u*^2 / sqrt(C_mu), epsilon = u*^3 / (kappa (z + z0)); u* and z0 fitted to the measured
  upwind wind profile of the experiment (``fit_log_law``).
- Floor and buildings: no-slip with standard wall functions. Sides and ceiling: slip.
- Outlet: fixed pressure, zero-gradient velocity.
- Mesh: blockMesh background (8 cm) refined to 2 cm at the buildings by snappyHexMesh.

The solved cell velocities are interpolated (nearest cell) to the 4 cm study grid and turned
into face velocities, then made exactly mass-consistent on that grid (a small correction:
the interpolation is not), so the same transport solver as L2/L3 can use them.

Runs in Docker (``opencfd/openfoam-default:2406``) on 4 cores.
"""

import subprocess
from pathlib import Path

import numpy as np

from tabpfn_spatial.study3d import physics as ph
from tabpfn_spatial.study3d.geometry import Grid, read_stl

IMAGE = "opencfd/openfoam-default:2406"
KAPPA, C_MU = 0.41, 0.09
DOMAIN = ((-2.4, -1.5, 0.0), (2.4, 1.5, 2.25))
BACKGROUND = 0.08


def fit_log_law(z: np.ndarray, u: np.ndarray, z_max: float = 0.8) -> tuple[float, float]:
    """(u*, z0) of U = u*/kappa ln(z / z0), least squares on z <= ``z_max`` (surface layer)."""
    m = (z > 0) & (z <= z_max) & (u > 0)
    a, b = np.polyfit(np.log(z[m]), u[m], 1)  # u = a ln z + b, a = u*/kappa
    ustar = a * KAPPA
    return float(ustar), float(np.exp(-b / a))


def write_stl_ascii(tri: np.ndarray, path: Path, name: str = "site") -> None:
    """ASCII STL without the floor plate (the domain floor is the ground)."""
    tri = tri[~(tri[:, :, 2] < 1e-3).all(1)]
    lines = [f"solid {name}"]
    for t in tri:
        n = np.cross(t[1] - t[0], t[2] - t[0])
        n = n / (np.linalg.norm(n) or 1.0)
        lines += [f"facet normal {n[0]:.6g} {n[1]:.6g} {n[2]:.6g}", "outer loop"]
        lines += [f"vertex {v[0]:.6g} {v[1]:.6g} {v[2]:.6g}" for v in t]
        lines += ["endloop", "endfacet"]
    lines.append(f"endsolid {name}")
    path.write_text("\n".join(lines) + "\n")


HEADER = """FoamFile {{ version 2.0; format ascii; class {cls}; object {obj}; }}
"""


def _write(case: Path, rel: str, cls: str, body: str) -> None:
    p = case / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(HEADER.format(cls=cls, obj=p.name) + body)


def write_case(case: Path, stl: Path, ustar: float, z0: float, iters: int = 1500) -> None:
    """All dictionaries of the OpenFOAM case in ``case``."""
    (lo, hi) = DOMAIN
    n = [round((h - l) / BACKGROUND) for l, h in zip(lo, hi, strict=True)]  # noqa: E741
    (case / "constant/triSurface").mkdir(parents=True, exist_ok=True)
    write_stl_ascii(read_stl(stl), case / "constant/triSurface/site.stl")
    uref, zref = ustar / KAPPA * np.log((0.5 + z0) / z0), 0.5
    _write(
        case,
        "system/blockMeshDict",
        "dictionary",
        f"""
convertToMeters 1;
vertices ( ({lo[0]} {lo[1]} {lo[2]}) ({hi[0]} {lo[1]} {lo[2]}) ({hi[0]} {hi[1]} {lo[2]})
  ({lo[0]} {hi[1]} {lo[2]}) ({lo[0]} {lo[1]} {hi[2]}) ({hi[0]} {lo[1]} {hi[2]})
  ({hi[0]} {hi[1]} {hi[2]}) ({lo[0]} {hi[1]} {hi[2]}) );
blocks ( hex (0 1 2 3 4 5 6 7) ({n[0]} {n[1]} {n[2]}) simpleGrading (1 1 1) );
boundary (
  inlet {{ type patch; faces ((1 2 6 5)); }}
  outlet {{ type patch; faces ((0 4 7 3)); }}
  floor {{ type wall; faces ((0 3 2 1)); }}
  top {{ type patch; faces ((4 5 6 7)); }}
  sides {{ type patch; faces ((0 1 5 4) (3 7 6 2)); }}
);
""",
    )
    _write(
        case,
        "system/snappyHexMeshDict",
        "dictionary",
        """
castellatedMesh true; snap true; addLayers false;
geometry { site.stl { type triSurfaceMesh; name site; }
  siteBox { type searchableBox; min (-1.3 -1.3 0); max (1.3 1.3 0.7); } }
castellatedMeshControls { maxLocalCells 2000000; maxGlobalCells 4000000; minRefinementCells 0;
  nCellsBetweenLevels 2; maxLoadUnbalance 0.1; resolveFeatureAngle 30;
  allowFreeStandingZoneFaces true;
  features ();
  refinementSurfaces { site { level (2 2); patchInfo { type wall; } } }
  refinementRegions { siteBox { mode inside; levels ((1E15 1)); } }
  locationInMesh (2.0 1.2 1.5); }
snapControls { nSmoothPatch 3; tolerance 2.0; nSolveIter 50; nRelaxIter 5; }
addLayersControls { relativeSizes true; layers {} expansionRatio 1; finalLayerThickness 0.3;
  minThickness 0.1; nGrow 0; featureAngle 60; nRelaxIter 3; nSmoothSurfaceNormals 1;
  nSmoothNormals 3; nSmoothThickness 10; maxFaceThicknessRatio 0.5;
  maxThicknessToMedialRatio 0.3; minMedianAxisAngle 90; nBufferCellsNoExtrude 0; nLayerIter 50; }
meshQualityControls { #includeEtc "caseDicts/meshQualityDict"
  nSmoothScale 4; errorReduction 0.75; relaxed { maxNonOrtho 75; } }
mergeTolerance 1e-6;
""",
    )
    _write(
        case,
        "system/controlDict",
        "dictionary",
        f"""
application simpleFoam; startFrom latestTime; startTime 0; stopAt endTime; endTime {iters};
deltaT 1; writeControl timeStep; writeInterval {iters}; purgeWrite 1; writeFormat ascii;
writePrecision 6; timeFormat general; runTimeModifiable true;
""",
    )
    _write(
        case,
        "system/fvSchemes",
        "dictionary",
        """
ddtSchemes { default steadyState; }
gradSchemes { default Gauss linear; grad(U) cellLimited Gauss linear 1; }
divSchemes { default none; div(phi,U) bounded Gauss linearUpwind grad(U);
  div(phi,k) bounded Gauss upwind; div(phi,epsilon) bounded Gauss upwind;
  div((nuEff*dev2(T(grad(U))))) Gauss linear; }
laplacianSchemes { default Gauss linear limited corrected 0.33; }
interpolationSchemes { default linear; }
snGradSchemes { default limited corrected 0.33; }
wallDist { method meshWave; }
""",
    )
    _write(
        case,
        "system/fvSolution",
        "dictionary",
        """
solvers { p { solver GAMG; smoother GaussSeidel; tolerance 1e-7; relTol 0.1; }
  "(U|k|epsilon)" { solver smoothSolver; smoother symGaussSeidel; tolerance 1e-8; relTol 0.1; } }
SIMPLE { nNonOrthogonalCorrectors 0; consistent true;
  residualControl { p 1e-4; U 1e-5; "(k|epsilon)" 1e-5; } }
relaxationFactors { equations { U 0.9; ".*" 0.7; } fields { p 1; } }
""",
    )
    _write(case, "system/decomposeParDict", "dictionary",
           "numberOfSubdomains 4; method scotch;\n")  # fmt: skip
    _write(case, "constant/transportProperties", "dictionary",
           "transportModel Newtonian; nu 1.5e-05;\n")  # fmt: skip
    _write(
        case,
        "constant/turbulenceProperties",
        "dictionary",
        "simulationType RAS;\nRAS { RASModel kEpsilon; turbulence on; printCoeffs on; }\n",
    )
    abl = f"""flowDir (-1 0 0); zDir (0 0 1); Uref {uref:.5g}; Zref {zref}; z0 uniform {z0:.5g};
    d uniform 0; kappa {KAPPA}; Cmu {C_MU};"""
    common = "sides { type slip; } top { type slip; }"
    _write(
        case,
        "0/U",
        "volVectorField",
        f"""
dimensions [0 1 -1 0 0 0 0]; internalField uniform (-{uref:.4g} 0 0);
boundaryField {{ inlet {{ type atmBoundaryLayerInletVelocity; {abl} value uniform (0 0 0); }}
  outlet {{ type inletOutlet; inletValue uniform (0 0 0); value uniform (0 0 0); }}
  floor {{ type noSlip; }} site {{ type noSlip; }} {common} }}
""",
    )
    _write(
        case,
        "0/p",
        "volScalarField",
        f"""
dimensions [0 2 -2 0 0 0 0]; internalField uniform 0;
boundaryField {{ inlet {{ type zeroGradient; }} outlet {{ type fixedValue; value uniform 0; }}
  floor {{ type zeroGradient; }} site {{ type zeroGradient; }} {common} }}
""",
    )
    k0 = ustar**2 / np.sqrt(C_MU)
    _write(
        case,
        "0/k",
        "volScalarField",
        f"""
dimensions [0 2 -2 0 0 0 0]; internalField uniform {k0:.5g};
boundaryField {{ inlet {{ type atmBoundaryLayerInletK; {abl} value uniform {k0:.5g}; }}
  outlet {{ type inletOutlet; inletValue uniform {k0:.5g}; value uniform {k0:.5g}; }}
  floor {{ type kqRWallFunction; value uniform {k0:.5g}; }}
  site {{ type kqRWallFunction; value uniform {k0:.5g}; }} {common} }}
""",
    )
    e0 = ustar**3 / (KAPPA * 0.5)
    _write(
        case,
        "0/epsilon",
        "volScalarField",
        f"""
dimensions [0 2 -3 0 0 0 0]; internalField uniform {e0:.5g};
boundaryField {{ inlet {{ type atmBoundaryLayerInletEpsilon; {abl} value uniform {e0:.5g}; }}
  outlet {{ type inletOutlet; inletValue uniform {e0:.5g}; value uniform {e0:.5g}; }}
  floor {{ type epsilonWallFunction; value uniform {e0:.5g}; }}
  site {{ type epsilonWallFunction; value uniform {e0:.5g}; }} {common} }}
""",
    )
    _write(
        case,
        "0/nut",
        "volScalarField",
        """
dimensions [0 2 -1 0 0 0 0]; internalField uniform 0;
boundaryField { inlet { type calculated; value uniform 0; }
  outlet { type calculated; value uniform 0; }
  floor { type nutkWallFunction; value uniform 0; } site { type nutkWallFunction; value uniform 0; }
  sides { type calculated; value uniform 0; } top { type calculated; value uniform 0; } }
""",
    )


SCRIPT = """source /usr/lib/openfoam/openfoam2406/etc/bashrc
set -e  # after the bashrc: it runs commands that return non-zero
cd /case
blockMesh > log.blockMesh 2>&1
snappyHexMesh -overwrite > log.snappy 2>&1
checkMesh > log.checkMesh 2>&1 || true
decomposePar -force > log.decompose 2>&1
mpirun --allow-run-as-root -np 4 simpleFoam -parallel > log.simpleFoam 2>&1
reconstructPar -latestTime > log.reconstruct 2>&1
postProcess -func writeCellCentres -latestTime > log.centres 2>&1
"""


def run_case(case: Path) -> None:
    """Mesh and solve in Docker (blocking; ~10-30 min on 4 cores)."""
    (case / "run.sh").write_text(SCRIPT)
    subprocess.run(["docker", "run", "--rm", "-v", f"{case.resolve()}:/case", IMAGE,
                    "bash", "/case/run.sh"], check=True)  # fmt: skip


def _read_field(path: Path, vector: bool) -> np.ndarray:
    """InternalField of an ASCII OpenFOAM volField (nonuniform list)."""
    txt = path.read_text()
    i = txt.index("internalField")
    j = txt.index("(", txt.index("List<", i))
    n = int(txt[txt.index(">", i) + 1 : j].strip())
    body = txt[j + 1 :]
    if vector:
        vals = np.fromstring(body.replace("(", " ").replace(")", " "), sep=" ", count=3 * n)
        return vals.reshape(n, 3)
    return np.fromstring(body, sep=" ", count=n)


def latest_time(case: Path) -> Path:
    """Directory of the last written time step."""
    times = [p for p in case.iterdir() if p.is_dir() and p.name.replace(".", "").isdigit()]
    return max(times, key=lambda p: float(p.name))


def cell_velocity(case: Path, g: Grid) -> np.ndarray:
    """CFD velocity at the study grid's cell centres (nearest CFD cell), (nx, ny, nz, 3)."""
    from scipy.spatial import cKDTree

    t = latest_time(case)
    u = _read_field(t / "U", vector=True)
    c = _read_field(t / "C", vector=True)
    xs, ys, zs = g.axis(0), g.axis(1), g.axis(2)
    centres = np.stack(np.meshgrid(xs, ys, zs, indexing="ij"), -1).reshape(-1, 3)
    _, idx = cKDTree(c).query(centres)
    return u[idx].reshape(*g.shape, 3)


def wind_on_grid(case: Path, g: Grid, solid: np.ndarray, flow: ph.Flow) -> ph.Faces:
    """CFD velocity on the study grid's faces, made mass-consistent there."""
    uc = cell_velocity(case, g)
    uc[solid] = 0.0
    ux = np.zeros((g.shape[0] + 1, *g.shape[1:]))
    uy = np.zeros((g.shape[0], g.shape[1] + 1, g.shape[2]))
    uz = np.zeros((*g.shape[:2], g.shape[2] + 1))
    ux[1:-1] = 0.5 * (uc[1:, ..., 0] + uc[:-1, ..., 0])
    ux[0], ux[-1] = uc[0, ..., 0], uc[-1, ..., 0]
    uy[:, 1:-1] = 0.5 * (uc[:, 1:, :, 1] + uc[:, :-1, :, 1])
    uy[:, 0], uy[:, -1] = uc[:, 0, :, 1], uc[:, -1, :, 1]
    uz[:, :, 1:-1] = 0.5 * (uc[:, :, 1:, 2] + uc[:, :, :-1, 2])
    uz[:, :, -1] = uc[:, :, -1, 2]
    for axis, arr in enumerate((ux, uy, uz)):
        arr[ph._blocked(solid, axis) | ph._closed_boundary(flow, g.shape)[axis]] = 0.0
    return ph.mass_consistent(g, solid, flow, ph.Faces(ux, uy, uz))


SC_T = 0.7  # turbulent Schmidt number (Tominaga & Stathopoulos 2007: 0.2-1.3, 0.7 typical)


def diffusivity_on_grid(case: Path, g: Grid) -> np.ndarray:
    """K = nu_t / Sc_t (m^2/s) from the CFD at the study grid's cell centres."""
    return scalar_on_grid(case, g, "nut") / SC_T


def scalar_on_grid(case: Path, g: Grid, name: str) -> np.ndarray:
    """A CFD scalar field (e.g. ``k``, ``nut``) at the study grid's cell centres (nearest)."""
    from scipy.spatial import cKDTree

    t = latest_time(case)
    k = _read_field(t / name, vector=False)
    c = _read_field(t / "C", vector=True)
    xs, ys, zs = g.axis(0), g.axis(1), g.axis(2)
    centres = np.stack(np.meshgrid(xs, ys, zs, indexing="ij"), -1).reshape(-1, 3)
    _, idx = cKDTree(c).query(centres)
    return k[idx].reshape(g.shape)
