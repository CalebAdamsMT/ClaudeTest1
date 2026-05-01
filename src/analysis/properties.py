"""Global engineering properties: center of gravity and inertia tensor."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class AnalysisResult:
    slices: list  # list[SliceResult]
    total_volume: float       # mm³
    total_mass: float         # kg
    center_of_gravity: np.ndarray  # shape (3,) in mm
    inertia_tensor: np.ndarray     # shape (3,3) in kg·mm²
    z_min: float
    z_max: float
    spacing_mm: float


def run_full_analysis(
    mesh,
    spacing_mm: float,
    density_kg_per_m3: float,
    progress_callback=None,
) -> AnalysisResult:
    """
    Top-level entry point called from the GUI worker thread.

    Parameters
    ----------
    mesh : trimesh.Trimesh
    spacing_mm : float
    density_kg_per_m3 : float
    progress_callback : callable(int), optional

    Returns
    -------
    AnalysisResult
    """
    from .slicer import compute_slices

    slices = compute_slices(mesh, spacing_mm, density_kg_per_m3, progress_callback)

    total_volume = sum(s.volume for s in slices)
    total_mass = sum(s.mass for s in slices)

    cog, inertia = _global_properties(mesh, density_kg_per_m3, slices, total_mass)

    return AnalysisResult(
        slices=slices,
        total_volume=total_volume,
        total_mass=total_mass,
        center_of_gravity=cog,
        inertia_tensor=inertia,
        z_min=float(mesh.bounds[0, 2]),
        z_max=float(mesh.bounds[1, 2]),
        spacing_mm=spacing_mm,
    )


def _global_properties(
    mesh,
    density_kg_per_m3: float,
    slices: list,
    total_mass: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (center_of_gravity [mm], inertia_tensor [kg·mm²])."""
    density_per_mm3 = density_kg_per_m3 * 1e-9

    if mesh.is_watertight:
        try:
            mesh.density = density_per_mm3
            cog = np.array(mesh.center_mass, dtype=float)
            inertia_origin = np.array(mesh.moment_inertia, dtype=float)
            # Shift inertia from origin to CoG via parallel axis theorem.
            mass = float(mesh.mass)
            d = cog
            d_sq = float(np.dot(d, d))
            inertia_cog = inertia_origin - mass * (d_sq * np.eye(3) - np.outer(d, d))
            return cog, inertia_cog
        except Exception:
            pass

    cog = _cog_from_slices(slices, total_mass)
    inertia = _inertia_from_slices(slices, cog, density_per_mm3)
    return cog, inertia


def _cog_from_slices(slices: list, total_mass: float) -> np.ndarray:
    if total_mass == 0:
        return np.zeros(3)

    cog = np.zeros(3)
    for s in slices:
        cog[0] += s.mass * s.centroid_x
        cog[1] += s.mass * s.centroid_y
        cog[2] += s.mass * s.z_mid
    return cog / total_mass


def _inertia_from_slices(
    slices: list,
    cog: np.ndarray,
    density_per_mm3: float,
) -> np.ndarray:
    """
    Accumulate the 3D inertia tensor about the global CoG from per-slice
    2D section properties using the parallel axis theorem.
    """
    I = np.zeros((3, 3))

    for s in slices:
        if s.mass == 0:
            continue

        h = s.z_top - s.z_bottom   # slice height (mm)
        m = s.mass                  # slice mass (kg)

        dx = s.centroid_x - cog[0]
        dy = s.centroid_y - cog[1]
        dz = s.z_mid - cog[2]

        # Centroidal 2D moment contributions scaled by density*height
        # give units of kg·mm²  (mm⁴ × kg/mm³ × mm = kg·mm²)
        ixx_vol = s.ixx * density_per_mm3 * h
        iyy_vol = s.iyy * density_per_mm3 * h
        ixy_vol = s.ixy * density_per_mm3 * h

        # Diagonal terms (parallel axis theorem: I_total = I_cm + m*d²)
        I[0, 0] += ixx_vol + m * (dy**2 + dz**2)
        I[1, 1] += iyy_vol + m * (dx**2 + dz**2)
        I[2, 2] += (ixx_vol + iyy_vol) + m * (dx**2 + dy**2)

        # Off-diagonal (products of inertia)
        I[0, 1] -= ixy_vol + m * dx * dy
        I[0, 2] -= m * dx * dz
        I[1, 2] -= m * dy * dz

    I[1, 0] = I[0, 1]
    I[2, 0] = I[0, 2]
    I[2, 1] = I[1, 2]

    return I
