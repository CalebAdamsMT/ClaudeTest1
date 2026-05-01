"""STEP file reader — converts .stp/.step to a trimesh.Trimesh and pyvista.PolyData."""

import numpy as np


def load_step(filepath: str):
    """
    Load a STEP file and return (trimesh.Trimesh, pyvista.PolyData).

    Raises
    ------
    FileNotFoundError
        If the file path does not exist.
    RuntimeError
        If the file cannot be parsed or produces no geometry.
    ImportError
        If pythonocc-core is not installed.
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
        from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_REVERSED
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
    shape = reader.OneShape()

    identity = gp_Trsf()
    builder = BRepBuilderAPI_Transform(shape, identity, True)
    flat_shape = builder.Shape()

    mesh_algo = BRepMesh_IncrementalMesh(flat_shape, 0.1, False, 0.5)
    mesh_algo.Perform()

    vertices_list = []
    faces_list = []
    vertex_offset = 0

    explorer = TopExp_Explorer(flat_shape, TopAbs_FACE)
    while explorer.More():
        face = explorer.Current()
        location = TopLoc_Location()
        triangulation = BRep_Tool.Triangulation(face, location)

        if triangulation is None:
            explorer.Next()
            continue

        n_nodes = triangulation.NbNodes()
        n_triangles = triangulation.NbTriangles()

        verts = np.array([
            [triangulation.Node(i).X(),
             triangulation.Node(i).Y(),
             triangulation.Node(i).Z()]
            for i in range(1, n_nodes + 1)
        ])

        is_reversed = (face.Orientation() == TopAbs_REVERSED)
        tris = []
        for j in range(1, n_triangles + 1):
            n1, n2, n3 = triangulation.Triangle(j).Get()
            if is_reversed:
                tris.append([n1 - 1 + vertex_offset, n3 - 1 + vertex_offset, n2 - 1 + vertex_offset])
            else:
                tris.append([n1 - 1 + vertex_offset, n2 - 1 + vertex_offset, n3 - 1 + vertex_offset])

        vertices_list.append(verts)
        faces_list.extend(tris)
        vertex_offset += n_nodes
        explorer.Next()

    if not vertices_list:
        raise RuntimeError("Tessellation produced no geometry. Try a different linear deflection value.")

    all_verts = np.vstack(vertices_list)
    all_faces = np.array(faces_list, dtype=np.int64)

    return _build_meshes(all_verts, all_faces)


def _build_meshes(vertices: np.ndarray, faces: np.ndarray):
    """Convert raw vertex/face arrays into trimesh.Trimesh and pyvista.PolyData."""
    import trimesh
    import pyvista

    tm = trimesh.Trimesh(vertices=vertices, faces=faces, process=True)

    face_count = np.full((len(tm.faces), 1), 3, dtype=np.int64)
    faces_pv = np.hstack([face_count, tm.faces]).ravel()
    pv_mesh = pyvista.PolyData(tm.vertices, faces_pv)

    return tm, pv_mesh
