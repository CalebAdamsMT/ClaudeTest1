"""Vertical sectioning of multiple bodies along the Z axis."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class BodySliceData:
    """Cross-section properties for a single body within one slab."""
    volume: float        # mm³ — volume of this body's slab
    mass: float          # kg
    area: float          # mm² — cross-section area at z_mid
    ixx: float           # mm⁴ — area MOI of this body's section about its centroid
    iyy: float
    ixy: float
    centroid_x: float    # mm — section centroid in model XY
    centroid_y: float


@dataclass
class SliceResult:
    index: int
    z_bottom: float
    z_top: float
    z_mid: float
    # Combined cross-section geometric properties (all bodies merged)
    area: float          # mm²
    ixx: float           # mm⁴
    iyy: float
    ixy: float
    centroid_x: float    # mm
    centroid_y: float
    # Combined totals
    volume: float        # mm³
    mass: float          # kg
    # Per-body breakdown
    bodies: list[BodySliceData] = field(default_factory=list)


def compute_slices(
    bodies: list[tuple],  # list of (trimesh.Trimesh, density_kg_per_m3)
    spacing_mm: float,
    progress_callback=None,
) -> list[SliceResult]:
    """
    Slice all bodies into horizontal layers spaced spacing_mm apart along Z.

    Parameters
    ----------
    bodies : list of (trimesh.Trimesh, float)
        Each tuple is a body mesh paired with its material density in kg/m³.
    spacing_mm : float
        Height of each slab in mm.
    progress_callback : callable(int), optional
        Progress percentage 0–100.

    Returns
    -------
    list[SliceResult]
    """
    # Z range spans all bodies.
    z_min = min(float(m.bounds[0, 2]) for m, _ in bodies)
    z_max = max(float(m.bounds[1, 2]) for m, _ in bodies)

    z_planes = np.arange(z_min, z_max, spacing_mm)
    if len(z_planes) < 2:
        z_planes = np.array([z_min, z_max])
    else:
        z_planes = np.append(z_planes, z_max)

    z_mids = (z_planes[:-1] + z_planes[1:]) / 2.0
    n = len(z_mids)

    # Precompute multiplane sections for every body.
    origins = np.column_stack([np.zeros(n), np.zeros(n), z_mids])
    normals = np.tile([0.0, 0.0, 1.0], (n, 1))

    body_sections = []  # list[list[Path3D or None]]
    for mesh, _ in bodies:
        secs = mesh.section_multiplane(plane_origins=origins, plane_normals=normals)
        body_sections.append(secs)

    results: list[SliceResult] = []

    for i, (z_lo, z_hi, z_mid) in enumerate(zip(z_planes[:-1], z_planes[1:], z_mids)):
        z_lo, z_hi, z_mid = float(z_lo), float(z_hi), float(z_mid)

        per_body: list[BodySliceData] = []
        all_polys_with_density = []

        for (mesh, density), secs in zip(bodies, body_sections):
            body_props = _section_properties(secs[i])
            volume = _slice_volume(mesh, z_lo, z_hi, body_props)
            density_mm3 = density * 1e-9
            mass = volume * density_mm3

            per_body.append(BodySliceData(
                volume=volume,
                mass=mass,
                area=body_props["area"],
                ixx=body_props["ixx"],
                iyy=body_props["iyy"],
                ixy=body_props["ixy"],
                centroid_x=body_props["centroid_x"],
                centroid_y=body_props["centroid_y"],
            ))

            if body_props["_polys"]:
                all_polys_with_density.extend(body_props["_polys"])

        # Combined cross-section properties from all bodies merged.
        combined = _combined_section_properties(all_polys_with_density)

        total_volume = sum(b.volume for b in per_body)
        total_mass = sum(b.mass for b in per_body)

        results.append(SliceResult(
            index=i,
            z_bottom=z_lo,
            z_top=z_hi,
            z_mid=z_mid,
            area=combined["area"],
            ixx=combined["ixx"],
            iyy=combined["iyy"],
            ixy=combined["ixy"],
            centroid_x=combined["centroid_x"],
            centroid_y=combined["centroid_y"],
            volume=total_volume,
            mass=total_mass,
            bodies=per_body,
        ))

        if progress_callback is not None:
            progress_callback(int((i + 1) / n * 100))

    return results


def _section_properties(section_3d) -> dict:
    """
    Extract 2D cross-section properties from a trimesh Path3D for one body.
    Returns a dict including '_polys' (list of shapely Polygons in model XY).
    """
    default = {
        "area": 0.0, "ixx": 0.0, "iyy": 0.0, "ixy": 0.0,
        "centroid_x": 0.0, "centroid_y": 0.0, "_polys": [],
    }

    if section_3d is None:
        return default

    try:
        section_2d, transform = section_3d.to_planar()
    except Exception:
        return default

    if section_2d is None:
        return default

    try:
        raw_polys = list(section_2d.polygons_full)
    except Exception:
        return default

    if not raw_polys:
        return default

    from shapely.ops import unary_union
    from shapely.geometry import Polygon
    from shapely.affinity import affine_transform

    valid = [p for p in raw_polys if p is not None and p.is_valid and p.area > 1e-12]
    if not valid:
        return default

    # Transform the 2D planar polygons back to model XY coordinates.
    # transform is a 4×4 matrix: planar (x,y,0,1) → model (X,Y,Z,1).
    # We only need the XY components (rows 0,1, cols 0,1,3).
    a, b, d, e = transform[0, 0], transform[0, 1], transform[1, 0], transform[1, 1]
    xoff, yoff = transform[0, 3], transform[1, 3]

    model_polys = []
    for p in valid:
        # shapely affine_transform uses [a,b,d,e,xoff,yoff] for 2D.
        mp = affine_transform(p, [a, b, d, e, xoff, yoff])
        if mp.is_valid and mp.area > 1e-12:
            model_polys.append(mp)

    if not model_polys:
        return default

    merged = unary_union(model_polys)
    if merged.is_empty:
        return default

    # Compute Ixx/Iyy for this body's section in model-space coordinates.
    sec_props = _compute_section_properties(merged)

    return {
        "area": sec_props["area"],
        "ixx": sec_props["ixx"],
        "iyy": sec_props["iyy"],
        "ixy": sec_props["ixy"],
        "centroid_x": sec_props["centroid_x"],
        "centroid_y": sec_props["centroid_y"],
        "_polys": model_polys,  # pass along for combined section calc
    }


def _combined_section_properties(polys) -> dict:
    """Compute section properties for all bodies' polygons merged together."""
    default = {"area": 0.0, "ixx": 0.0, "iyy": 0.0, "ixy": 0.0,
               "centroid_x": 0.0, "centroid_y": 0.0}

    if not polys:
        return default

    from shapely.ops import unary_union

    merged = unary_union(polys)
    if merged.is_empty:
        return default

    return _compute_section_properties(merged)


def _compute_section_properties(shapely_geom) -> dict:
    """Run sectionproperties on a shapely geometry (Polygon or MultiPolygon)."""
    default = {"area": 0.0, "ixx": 0.0, "iyy": 0.0, "ixy": 0.0,
               "centroid_x": 0.0, "centroid_y": 0.0}
    try:
        from shapely.geometry import Polygon, MultiPolygon
        from sectionproperties.pre.geometry import Geometry, CompoundGeometry
        from sectionproperties.analysis import Section

        if isinstance(shapely_geom, Polygon):
            geom = Geometry(shapely_geom)
        else:
            sub = [Geometry(p) for p in shapely_geom.geoms if p.area > 1e-12]
            if not sub:
                return default
            geom = CompoundGeometry(sub)

        mesh_size = max(shapely_geom.area / 200.0, 0.01)
        geom = geom.create_mesh(mesh_sizes=[mesh_size])
        sec = Section(geometry=geom)
        sec.calculate_geometric_properties()

        ixx_c, iyy_c, ixy_c = sec.get_ic()
        cx, cy = sec.get_c()
        area = sec.get_area()

        return {
            "area": float(area),
            "ixx": float(ixx_c),
            "iyy": float(iyy_c),
            "ixy": float(ixy_c),
            "centroid_x": float(cx),
            "centroid_y": float(cy),
        }
    except Exception:
        return default


def _slice_volume(mesh, z_lo: float, z_hi: float, props: dict) -> float:
    """Volume of a single body's slab between z_lo and z_hi."""
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

    # Fallback: area × height.
    return props["area"] * (z_hi - z_lo)
