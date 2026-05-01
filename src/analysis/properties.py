"""Global engineering properties: center of gravity and inertia tensor."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class AnalysisResult:
    slices: list                           # list[SliceResult]
    total_volume: float                    # mm³  (all bodies)
    total_mass: float                      # kg   (all bodies)
    body_volumes: list[float]              # mm³ per body
    body_masses: list[float]               # kg  per body
    body_densities: list[float]            # kg/m³ per body
    center_of_gravity: np.ndarray          # shape (3,) in mm
    inertia_tensor: np.ndarray             # shape (3,3) in kg·mm²
    z_min: float
    z_max: float
    spacing_mm: float
    n_bodies: int


def run_full_analysis(
    bodies: list[tuple],   # list of (trimesh.Trimesh, density_kg_per_m3)
    spacing_mm: float,
    progress_callback=None,
) -> AnalysisResult:
    """
    Top-level entry point called from the GUI worker thread.

    Parameters
    ----------
    bodies : list of (trimesh.Trimesh, float)
    spacing_mm : float
    progress_callback : callable(int), optional

    Returns
    -------
    AnalysisResult
    """
    from .slicer import compute_slices

    slices = compute_slices(bodies, spacing_mm, progress_callback)

    densities = [d for _, d in bodies]
    body_volumes = [sum(s.bodies[i].volume for s in slices) for i in range(len(bodies))]
    body_masses = [sum(s.bodies[i].mass for s in slices) for i in range(len(bodies))]
    total_volume = sum(body_volumes)
    total_mass = sum(body_masses)

    cog, inertia = _global_properties(bodies, slices, densities, total_mass)

    meshes = [m for m, _ in bodies]
    z_min = min(float(m.bounds[0, 2]) for m in meshes)
    z_max = max(float(m.bounds[1, 2]) for m in meshes)

    return AnalysisResult(
        slices=slices,
        total_volume=total_volume,
        total_mass=total_mass,
        body_volumes=body_volumes,
        body_masses=body_masses,
        body_densities=densities,
        center_of_gravity=cog,
        inertia_tensor=inertia,
        z_min=z_min,
        z_max=z_max,
        spacing_mm=spacing_mm,
        n_bodies=len(bodies),
    )


def _global_properties(
    bodies: list[tuple],
    slices: list,
    densities: list[float],
    total_mass: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (center_of_gravity [mm], inertia_tensor [kg·mm²])."""

    # Try trimesh mass properties if every body is watertight.
    if all(m.is_watertight for m, _ in bodies):
        try:
            return _properties_from_meshes(bodies)
        except Exception:
            pass

    cog = _cog_from_slices(slices, total_mass)
    inertia = _inertia_from_slices(slices, cog, densities)
    return cog, inertia


def _properties_from_meshes(bodies: list[tuple]) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute combined CoG and inertia tensor from individual watertight body meshes.
    Each body is processed independently then combined via parallel axis theorem.
    """
    body_props = []
    for mesh, density in bodies:
        mesh.density = density * 1e-9  # kg/mm³
        mass = float(mesh.mass)
        cog_i = np.array(mesh.center_mass, dtype=float)
        # trimesh.moment_inertia is about the mesh origin.
        I_origin = np.array(mesh.moment_inertia, dtype=float)
        # Shift to body CoG.
        d = cog_i
        I_cog_i = I_origin - mass * (np.dot(d, d) * np.eye(3) - np.outer(d, d))
        body_props.append((mass, cog_i, I_cog_i))

    total_mass = sum(m for m, _, _ in body_props)
    if total_mass == 0:
        return np.zeros(3), np.zeros((3, 3))

    # Combined CoG.
    cog = sum(m * c for m, c, _ in body_props) / total_mass

    # Combined inertia at overall CoG via parallel axis theorem.
    I_total = np.zeros((3, 3))
    for mass_i, cog_i, I_cog_i in body_props:
        d = cog_i - cog
        d_sq = np.dot(d, d)
        I_total += I_cog_i + mass_i * (d_sq * np.eye(3) - np.outer(d, d))

    return cog, I_total


def _cog_from_slices(slices: list, total_mass: float) -> np.ndarray:
    if total_mass == 0:
        return np.zeros(3)

    cog = np.zeros(3)
    for s in slices:
        for b in s.bodies:
            cog[0] += b.mass * b.centroid_x
            cog[1] += b.mass * b.centroid_y
        cog[2] += s.mass * s.z_mid
    return cog / total_mass


def _inertia_from_slices(
    slices: list,
    cog: np.ndarray,
    densities: list[float],
) -> np.ndarray:
    """
    Accumulate 3D inertia tensor about the overall CoG.

    Each body's contribution to each slab is weighted by its own density,
    using per-body area moment of inertia values from the sectionproperties analysis.
    """
    I = np.zeros((3, 3))

    for s in slices:
        h = s.z_top - s.z_bottom
        dz = s.z_mid - cog[2]

        for bi, (b, density) in enumerate(zip(s.bodies, densities)):
            if b.mass == 0:
                continue

            density_mm3 = density * 1e-9
            m = b.mass
            dx = b.centroid_x - cog[0]
            dy = b.centroid_y - cog[1]

            # Mass second-moment contributions from this body's cross-section.
            # Units: mm⁴ × kg/mm³ × mm = kg·mm²
            ixx_vol = b.ixx * density_mm3 * h
            iyy_vol = b.iyy * density_mm3 * h
            ixy_vol = b.ixy * density_mm3 * h

            I[0, 0] += ixx_vol + m * (dy**2 + dz**2)
            I[1, 1] += iyy_vol + m * (dx**2 + dz**2)
            I[2, 2] += (ixx_vol + iyy_vol) + m * (dx**2 + dy**2)

            I[0, 1] -= ixy_vol + m * dx * dy
            I[0, 2] -= m * dx * dz
            I[1, 2] -= m * dy * dz

    I[1, 0] = I[0, 1]
    I[2, 0] = I[0, 2]
    I[2, 1] = I[1, 2]

    return I
