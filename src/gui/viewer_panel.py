"""3D viewer panel — embeds a PyVista QtInteractor inside a QWidget."""

from __future__ import annotations

import numpy as np
from PyQt5.QtWidgets import QWidget, QVBoxLayout


class ViewerPanel(QWidget):
    """Embeds a PyVista interactive 3D renderer into the Qt layout."""

    def __init__(self, parent=None):
        super().__init__(parent)
        from pyvistaqt import QtInteractor
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.plotter = QtInteractor(self)
        self.plotter.set_background("gray15")
        layout.addWidget(self.plotter.interactor)

        self._mesh_actor = None
        self._plane_actors: list = []
        self._highlight_actor = None
        self._current_pv_mesh = None

    def display_mesh(self, pv_mesh) -> None:
        """Render the loaded mesh."""
        self.plotter.clear()
        self._plane_actors.clear()
        self._highlight_actor = None

        self._current_pv_mesh = pv_mesh
        self._mesh_actor = self.plotter.add_mesh(
            pv_mesh,
            color="lightsteelblue",
            opacity=0.85,
            show_edges=False,
            smooth_shading=True,
        )
        self.plotter.reset_camera()
        self.plotter.render()

    def draw_section_planes(self, result) -> None:
        """Draw semi-transparent cutting planes at each section boundary."""
        for actor in self._plane_actors:
            self.plotter.remove_actor(actor)
        self._plane_actors.clear()

        if self._current_pv_mesh is None:
            return

        bounds = self._current_pv_mesh.bounds  # (xmin, xmax, ymin, ymax, zmin, zmax)
        width = max(bounds[1] - bounds[0], bounds[3] - bounds[2]) * 1.1
        cx = (bounds[0] + bounds[1]) / 2.0
        cy = (bounds[2] + bounds[3]) / 2.0

        z_levels = set()
        for s in result.slices:
            z_levels.add(s.z_bottom)
            z_levels.add(s.z_top)

        for z in z_levels:
            import pyvista
            plane = pyvista.Plane(
                center=(cx, cy, z),
                direction=(0, 0, 1),
                i_size=width,
                j_size=width,
                i_resolution=1,
                j_resolution=1,
            )
            actor = self.plotter.add_mesh(
                plane,
                color="cyan",
                opacity=0.25,
                style="surface",
            )
            self._plane_actors.append(actor)

        self.plotter.render()

    def highlight_slice(self, slice_result) -> None:
        """Highlight a specific slab in the viewer."""
        if self._current_pv_mesh is None:
            return

        if self._highlight_actor is not None:
            self.plotter.remove_actor(self._highlight_actor)
            self._highlight_actor = None

        try:
            clipped = self._current_pv_mesh.clip(
                normal=(0, 0, -1), origin=(0, 0, slice_result.z_top)
            )
            clipped = clipped.clip(
                normal=(0, 0, 1), origin=(0, 0, slice_result.z_bottom)
            )
            self._highlight_actor = self.plotter.add_mesh(
                clipped,
                color="orange",
                opacity=0.8,
            )
        except Exception:
            pass

        self.plotter.render()

    def clear_highlight(self) -> None:
        if self._highlight_actor is not None:
            self.plotter.remove_actor(self._highlight_actor)
            self._highlight_actor = None
            self.plotter.render()

    def close_plotter(self) -> None:
        """Must be called on window close to properly shut down VTK."""
        self.plotter.close()
