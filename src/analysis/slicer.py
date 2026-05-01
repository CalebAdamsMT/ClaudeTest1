"""Vertical sectioning of a trimesh along the Z axis."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class SliceResult:
    index: int
    z_bottom: float
    z_top: float
    z_mid: float
    area: float           # cross-section area at z_mid (mm²)
    ixx: float            # area moment of inertia about x-axis through centroid (mm⁴)
    iyy: float            # area moment of inertia about y-axis through centroid (mm⁴)
    ixy: float            # product of inertia (mm⁴)
    centroid_x: float     # centroid x in model XY plane (mm)
    centroid_y: float     # centroid y in model XY plane (mm)
    volume: float         # volume between z_bottom and z_top (mm³)
    mass: float           # volume * density (kg)


def compute_slices(
    mesh,
    spacing_mm: float,
    density_kg_per_m3: float,
    progress_callback=None,
) -> list[SliceResult]:
    """
    Slice the mesh into horizontal layers spaced spacing_mm apart along Z.

    Parameters
    ----------
    mesh : trimesh.Trimesh
    spacing_mm : float
        Distance between section planes in mm.
    density_kg_per_m3 : float
        Material density in kg/m³.
    progress_callback : callable(int), optional
        Called with a progress percentage (0-100) as each section completes.

    Returns
    -------
    list[SliceResult]
        One entry per slab between adjacent Z planes.
    """
    z_min = float(mesh.bounds[0, 2])
    z_max = float(mesh.bounds[1, 2])

    z_planes = np.arange(z_min, z_max, spacing_mm)
    if len(z_planes) < 2:
        z_planes = np.array([z_min, z_max])
    else:
        z_planes = np.append(z_planes, z_max)

    z_mids = (z_planes[:-1] + z_planes[1:]) / 2.0

    origins = np.column_stack([
        np.zeros(len(z_mids)),
        np.zeros(len(z_mids)),
        z_mids,
    ])
    normals = np.tile([0.0, 0.0, 1.0], (len(z_mids), 1))

    sections_3d = mesh.section_multiplane(
        plane_origins=origins,
        plane_normals=normals,
    )

    results: list[SliceResult] = []
    n = len(z_mids)

    for i, (z_lo, z_hi, z_mid, section_3d) in enumerate(
        zip(z_planes[:-1], z_planes[1:], z_mids, sections_3d)
    ):
        props = _section_properties(section_3d)
        volume = _slice_volume(mesh, float(z_lo), float(z_hi), props)
        density_per_mm3 = density_kg_per_m3 * 1e-9
        mass = volume * density_per_mm3

        results.append(SliceResult(
            index=i,
            z_bottom=float(z_lo),
            z_top=float(z_hi),
            z_mid=float(z_mid),
            area=props["area"],
            ixx=props["ixx"],
            iyy=props["iyy"],
            ixy=props["ixy"],
            centroid_x=props["centroid_x"],
            centroid_y=props["centroid_y"],
            volume=volume,
            mass=mass,
        ))

        if progress_callback is not None:
            progress_callback(int((i + 1) / n * 100))

    return results


def _section_properties(section_3d) -> dict:
    """Extract 2D cross-section properties from a trimesh Path3D."""
    default = {"area": 0.0, "ixx": 0.0, "iyy": 0.0, "ixy": 0.0,
               "centroid_x": 0.0, "centroid_y": 0.0}

    if section_3d is None:
        return default

    try:
        section_2d, transform = section_3d.to_planar()
    except Exception:
        return default

    if section_2d is None:
        return default

    try:
        polys = list(section_2d.polygons_full)
    except Exception:
        return default

    if not polys:
        return default

    try:
        from shapely.ops import unary_union
        from shapely.geometry import MultiPolygon, Polygon

        valid_polys = [p for p in polys if p is not None and p.is_valid and p.area > 1e-12]
        if not valid_polys:
            return default

        merged = unary_union(valid_polys)
        if merged.is_empty:
            return default

        from sectionproperties.pre.geometry import Geometry, CompoundGeometry
        from sectionproperties.analysis import Section

        if isinstance(merged, Polygon):
            geom = Geometry(merged)
        else:
            geom = CompoundGeometry([Geometry(p) for p in merged.geoms if p.area > 1e-12])

        mesh_size = max(merged.area / 200.0, 0.01)
        geom = geom.create_mesh(mesh_sizes=[mesh_size])
        sec = Section(geometry=geom)
        sec.calculate_geometric_properties()

        ixx_c, iyy_c, ixy_c = sec.get_ic()
        cx, cy = sec.get_c()
        area = sec.get_area()

        # Transform centroid back to 3D model XY space.
        # transform is a 4x4 matrix mapping planar → 3D.
        centroid_local = np.array([cx, cy, 0.0, 1.0])
        centroid_3d = transform @ centroid_local

        return {
            "area": float(area),
            "ixx": float(ixx_c),
            "iyy": float(iyy_c),
            "ixy": float(ixy_c),
            "centroid_x": float(centroid_3d[0]),
            "centroid_y": float(centroid_3d[1]),
        }
    except Exception:
        return default


def _slice_volume(mesh, z_lo: float, z_hi: float, props: dict) -> float:
    """
    Estimate the volume of the mesh slab between z_lo and z_hi.

    For watertight meshes, clips and computes volume directly.
    Falls back to area * height integration.
    """
    if mesh.is_watertight:
        try:
            import pyvista
            face_count = np.full((len(mesh.faces), 1), 3, dtype=np.int64)
            faces_pv = np.hstack([face_count, mesh.faces]).ravel()
            pv_mesh = pyvista.PolyData(mesh.vertices, faces_pv)

            clipped = pv_mesh.clip(normal=(0, 0, -1), origin=(0, 0, z_hi))
            clipped = clipped.clip(normal=(0, 0, 1), origin=(0, 0, z_lo))
            filled = clipped.fill_holes(hole_size=1e10)
            return float(filled.volume)
        except Exception:
            pass

    return props["area"] * (z_hi - z_lo)
