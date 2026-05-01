"""STEP file reader — returns one trimesh.Trimesh per solid body."""

import numpy as np


# Colors assigned to each body in the viewer (CSS names pyvista understands).
BODY_COLORS = ["lightsteelblue", "lightsalmon", "lightgreen", "plum", "khaki"]


def load_step(filepath: str):
    """
    Load a STEP file and return (trimesh_list, pyvista_list).

    Each element corresponds to one solid body found in the file.

    Returns
    -------
    trimesh_list : list[trimesh.Trimesh]
    pyvista_list : list[pyvista.PolyData]

    Raises
    ------
    FileNotFoundError
    RuntimeError
    ImportError
    """
    import os
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    try:
        from OCC.Core.STEPControl import STEPControl_Reader
        from OCC.Core.IFSelect import IFSelect_RetDone
        from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Transform
        from OCC.Core.gp import gp_Trsf
        from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
        from OCC.Core.TopExp import TopExp_Explorer
        from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_REVERSED, TopAbs_SOLID, TopAbs_SHELL
        from OCC.Core.BRep import BRep_Tool
        from OCC.Core.TopLoc import TopLoc_Location
    except ImportError as exc:
        raise ImportError(
            "pythonocc-core is required to read STEP files. "
            "Install it with: conda install -c conda-forge pythonocc-core"
        ) from exc

    reader = STEPControl_Reader()
    status = reader.ReadFile(filepath)
    if status != IFSelect_RetDone:
        raise RuntimeError(f"Failed to parse STEP file: {filepath}")

    n_roots = reader.NbRootsForTransfer()
    if n_roots == 0:
        raise RuntimeError(f"STEP file contains no transferable entities: {filepath}")

    reader.TransferRoots()
    compound = reader.OneShape()

    # Try to find individual solid bodies; fall back to shells, then the whole compound.
    solids = _explore_shapes(compound, TopAbs_SOLID)
    if not solids:
        solids = _explore_shapes(compound, TopAbs_SHELL)
    if not solids:
        solids = [compound]

    trimesh_list = []
    pyvista_list = []

    for solid in solids:
        try:
            tm, pv = _tessellate_shape(solid)
        except Exception:
            continue
        if len(tm.faces) == 0:
            continue
        trimesh_list.append(tm)
        pyvista_list.append(pv)

    if not trimesh_list:
        raise RuntimeError(
            "Tessellation produced no geometry. The file may be empty or corrupt."
        )

    return trimesh_list, pyvista_list


def _explore_shapes(parent_shape, shape_type) -> list:
    from OCC.Core.TopExp import TopExp_Explorer
    shapes = []
    explorer = TopExp_Explorer(parent_shape, shape_type)
    while explorer.More():
        shapes.append(explorer.Current())
        explorer.Next()
    return shapes


def _tessellate_shape(shape):
    """Tessellate a single OCC shape and return (trimesh.Trimesh, pyvista.PolyData)."""
    from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Transform
    from OCC.Core.gp import gp_Trsf
    from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
    from OCC.Core.TopExp import TopExp_Explorer
    from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_REVERSED
    from OCC.Core.BRep import BRep_Tool
    from OCC.Core.TopLoc import TopLoc_Location

    # Flatten all sub-shape locations to identity to avoid transform extraction issues.
    identity = gp_Trsf()
    builder = BRepBuilderAPI_Transform(shape, identity, True)
    flat = builder.Shape()

    mesh_algo = BRepMesh_IncrementalMesh(flat, 0.1, False, 0.5)
    mesh_algo.Perform()

    vertices_list = []
    faces_list = []
    vertex_offset = 0

    explorer = TopExp_Explorer(flat, TopAbs_FACE)
    while explorer.More():
        face = explorer.Current()
        loc = TopLoc_Location()
        tri = BRep_Tool.Triangulation(face, loc)

        if tri is None:
            explorer.Next()
            continue

        n_nodes = tri.NbNodes()
        n_tris = tri.NbTriangles()

        verts = np.array([
            [tri.Node(i).X(), tri.Node(i).Y(), tri.Node(i).Z()]
            for i in range(1, n_nodes + 1)
        ])

        is_reversed = (face.Orientation() == TopAbs_REVERSED)
        tris = []
        for j in range(1, n_tris + 1):
            n1, n2, n3 = tri.Triangle(j).Get()
            if is_reversed:
                tris.append([n1 - 1 + vertex_offset, n3 - 1 + vertex_offset, n2 - 1 + vertex_offset])
            else:
                tris.append([n1 - 1 + vertex_offset, n2 - 1 + vertex_offset, n3 - 1 + vertex_offset])

        vertices_list.append(verts)
        faces_list.extend(tris)
        vertex_offset += n_nodes
        explorer.Next()

    if not vertices_list:
        raise RuntimeError("No triangulated faces found on this shape.")

    all_verts = np.vstack(vertices_list)
    all_faces = np.array(faces_list, dtype=np.int64)

    return _build_meshes(all_verts, all_faces)


def _build_meshes(vertices: np.ndarray, faces: np.ndarray):
    import trimesh
    import pyvista

    tm = trimesh.Trimesh(vertices=vertices, faces=faces, process=True)

    face_count = np.full((len(tm.faces), 1), 3, dtype=np.int64)
    faces_pv = np.hstack([face_count, tm.faces]).ravel()
    pv_mesh = pyvista.PolyData(tm.vertices, faces_pv)

    return tm, pv_mesh
