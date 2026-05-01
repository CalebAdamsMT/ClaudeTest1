"""3D viewer panel — embeds a PyVista QtInteractor inside a QWidget."""

from __future__ import annotations

import numpy as np
from PyQt5.QtWidgets import QWidget, QVBoxLayout

# Must stay in sync with BODY_COLORS in step_reader.py.
_BODY_COLORS = ["lightsteelblue", "lightsalmon", "lightgreen", "plum", "khaki"]


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

        self._body_actors: list = []
        self._plane_actors: list = []
        self._highlight_actors: list = []
        self._current_pv_meshes: list = []  # one pyvista.PolyData per body

    def display_bodies(self, pv_meshes: list) -> None:
        """Render all body meshes, each in a distinct color."""
        self.plotter.clear()
        self._body_actors.clear()
        self._plane_actors.clear()
        self._highlight_actors.clear()
        self._current_pv_meshes = list(pv_meshes)

        for i, pv_mesh in enumerate(pv_meshes):
            color = _BODY_COLORS[i % len(_BODY_COLORS)]
            actor = self.plotter.add_mesh(
                pv_mesh,
                color=color,
                opacity=0.85,
                show_edges=False,
                smooth_shading=True,
                label=f"Body {i + 1}",
            )
            self._body_actors.append(actor)

        self.plotter.reset_camera()
        self.plotter.render()

    def draw_section_planes(self, result) -> None:
        """Draw semi-transparent cutting planes at each section boundary."""
        for actor in self._plane_actors:
            self.plotter.remove_actor(actor)
        self._plane_actors.clear()

        if not self._current_pv_meshes:
            return

        # Compute a bounding box that spans all bodies.
        all_bounds = [m.bounds for m in self._current_pv_meshes]
        xmin = min(b[0] for b in all_bounds)
        xmax = max(b[1] for b in all_bounds)
        ymin = min(b[2] for b in all_bounds)
        ymax = max(b[3] for b in all_bounds)

        width = max(xmax - xmin, ymax - ymin) * 1.1
        cx = (xmin + xmax) / 2.0
        cy = (ymin + ymax) / 2.0

        z_levels: set[float] = set()
        for s in result.slices:
            z_levels.add(s.z_bottom)
            z_levels.add(s.z_top)

        import pyvista
        for z in sorted(z_levels):
            plane = pyvista.Plane(
                center=(cx, cy, z),
                direction=(0, 0, 1),
                i_size=width,
                j_size=width,
                i_resolution=1,
                j_resolution=1,
            )
            actor = self.plotter.add_mesh(plane, color="cyan", opacity=0.25, style="surface")
            self._plane_actors.append(actor)

        self.plotter.render()

    def highlight_slice(self, slice_result) -> None:
        """Highlight a specific slab across all bodies."""
        for actor in self._highlight_actors:
            self.plotter.remove_actor(actor)
        self._highlight_actors.clear()

        for pv_mesh in self._current_pv_meshes:
            try:
                clipped = pv_mesh.clip(normal=(0, 0, -1), origin=(0, 0, slice_result.z_top))
                clipped = clipped.clip(normal=(0, 0, 1), origin=(0, 0, slice_result.z_bottom))
                actor = self.plotter.add_mesh(clipped, color="yellow", opacity=0.9)
                self._highlight_actors.append(actor)
            except Exception:
                pass

        self.plotter.render()

    def clear_highlight(self) -> None:
        for actor in self._highlight_actors:
            self.plotter.remove_actor(actor)
        self._highlight_actors.clear()
        self.plotter.render()

    def close_plotter(self) -> None:
        """Must be called on window close to properly shut down VTK."""
        self.plotter.close()
