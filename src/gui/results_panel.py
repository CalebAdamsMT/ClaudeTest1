"""Results panel — shows per-section properties and global summary."""

from __future__ import annotations

import csv

import numpy as np
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)


_COLUMNS = [
    ("Section", ""),
    ("Z Bottom", "mm"),
    ("Z Top", "mm"),
    ("Volume", "mm³"),
    ("Mass", "kg"),
    ("Area", "mm²"),
    ("Ixx", "mm⁴"),
    ("Iyy", "mm⁴"),
    ("Ixy", "mm⁴"),
    ("Centroid X", "mm"),
    ("Centroid Y", "mm"),
]


class ResultsPanel(QWidget):
    """Displays a table of per-section results and an overall summary."""

    row_selected = pyqtSignal(int)  # emits slice index when a row is clicked

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        self._summary_group = QGroupBox("Overall Properties")
        summary_layout = QVBoxLayout(self._summary_group)
        self._summary_label = QLabel("Load a file and run analysis to see results.")
        self._summary_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self._summary_label.setWordWrap(True)
        summary_layout.addWidget(self._summary_label)
        layout.addWidget(self._summary_group)

        self._table = QTableWidget()
        self._table.setColumnCount(len(_COLUMNS))
        self._table.setHorizontalHeaderLabels(
            [f"{name}\n({unit})" if unit else name for name, unit in _COLUMNS]
        )
        self._table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._table.setSelectionBehavior(QTableWidget.SelectRows)
        self._table.setAlternatingRowColors(True)
        self._table.horizontalHeader().setStretchLastSection(True)
        self._table.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._table.itemSelectionChanged.connect(self._on_selection_changed)
        layout.addWidget(self._table)

        self._result = None

    def populate(self, result) -> None:
        """Fill the table and summary from an AnalysisResult."""
        self._result = result
        self._fill_summary(result)
        self._fill_table(result)

    def _fill_summary(self, result) -> None:
        cog = result.center_of_gravity
        I = result.inertia_tensor
        text = (
            f"<b>Total Volume:</b> {result.total_volume:.4f} mm³<br>"
            f"<b>Total Mass:</b> {result.total_mass:.6f} kg<br>"
            f"<b>Center of Gravity:</b> "
            f"X={cog[0]:.4f} mm, Y={cog[1]:.4f} mm, Z={cog[2]:.4f} mm<br>"
            f"<b>Inertia Tensor about CoG (kg·mm²):</b><br>"
            f"&nbsp;&nbsp;[{I[0,0]:.4e}, {I[0,1]:.4e}, {I[0,2]:.4e}]<br>"
            f"&nbsp;&nbsp;[{I[1,0]:.4e}, {I[1,1]:.4e}, {I[1,2]:.4e}]<br>"
            f"&nbsp;&nbsp;[{I[2,0]:.4e}, {I[2,1]:.4e}, {I[2,2]:.4e}]"
        )
        self._summary_label.setText(text)

    def _fill_table(self, result) -> None:
        self._table.setRowCount(len(result.slices))
        for row, s in enumerate(result.slices):
            values = [
                str(s.index + 1),
                f"{s.z_bottom:.4f}",
                f"{s.z_top:.4f}",
                f"{s.volume:.4f}",
                f"{s.mass:.6f}",
                f"{s.area:.4f}",
                f"{s.ixx:.4e}",
                f"{s.iyy:.4e}",
                f"{s.ixy:.4e}",
                f"{s.centroid_x:.4f}",
                f"{s.centroid_y:.4f}",
            ]
            for col, val in enumerate(values):
                item = QTableWidgetItem(val)
                item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                self._table.setItem(row, col, item)

        self._table.resizeColumnsToContents()

    def _on_selection_changed(self) -> None:
        selected = self._table.selectedItems()
        if selected:
            row = self._table.row(selected[0])
            self.row_selected.emit(row)

    def export_csv(self, path: str) -> None:
        """Write the section results and global summary to a CSV file."""
        if self._result is None:
            return

        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)

            # Overall summary
            cog = self._result.center_of_gravity
            I = self._result.inertia_tensor
            writer.writerow(["OVERALL PROPERTIES"])
            writer.writerow(["Total Volume (mm³)", f"{self._result.total_volume:.6f}"])
            writer.writerow(["Total Mass (kg)", f"{self._result.total_mass:.6f}"])
            writer.writerow(["CoG X (mm)", f"{cog[0]:.6f}"])
            writer.writerow(["CoG Y (mm)", f"{cog[1]:.6f}"])
            writer.writerow(["CoG Z (mm)", f"{cog[2]:.6f}"])
            writer.writerow(["Ixx about CoG (kg·mm²)", f"{I[0,0]:.6e}"])
            writer.writerow(["Iyy about CoG (kg·mm²)", f"{I[1,1]:.6e}"])
            writer.writerow(["Izz about CoG (kg·mm²)", f"{I[2,2]:.6e}"])
            writer.writerow(["Ixy about CoG (kg·mm²)", f"{I[0,1]:.6e}"])
            writer.writerow(["Ixz about CoG (kg·mm²)", f"{I[0,2]:.6e}"])
            writer.writerow(["Iyz about CoG (kg·mm²)", f"{I[1,2]:.6e}"])
            writer.writerow([])

            # Section table
            header = [f"{name} ({unit})" if unit else name for name, unit in _COLUMNS]
            writer.writerow(header)
            for s in self._result.slices:
                writer.writerow([
                    s.index + 1,
                    f"{s.z_bottom:.6f}",
                    f"{s.z_top:.6f}",
                    f"{s.volume:.6f}",
                    f"{s.mass:.8f}",
                    f"{s.area:.6f}",
                    f"{s.ixx:.6e}",
                    f"{s.iyy:.6e}",
                    f"{s.ixy:.6e}",
                    f"{s.centroid_x:.6f}",
                    f"{s.centroid_y:.6f}",
                ])

    def clear(self) -> None:
        self._table.setRowCount(0)
        self._summary_label.setText("Load a file and run analysis to see results.")
        self._result = None
