"""Main application window."""

from __future__ import annotations

import os

from PyQt5.QtCore import QThread, Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QAction,
    QDoubleSpinBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

from .viewer_panel import ViewerPanel, _BODY_COLORS
from .results_panel import ResultsPanel

# Default densities to pre-fill for body 1 and 2 (steel, aluminium).
_DEFAULT_DENSITIES = [7800.0, 2700.0, 1000.0, 1000.0, 1000.0]


class _LoadWorker(QThread):
    """Loads and tessellates a STEP file off the main thread."""

    finished = pyqtSignal(object, object)  # (list[trimesh], list[pyvista.PolyData])
    error = pyqtSignal(str)

    def __init__(self, filepath: str, parent=None):
        super().__init__(parent)
        self._filepath = filepath

    def run(self) -> None:
        try:
            from src.io.step_reader import load_step
            tm_list, pv_list = load_step(self._filepath)
            self.finished.emit(tm_list, pv_list)
        except Exception as exc:
            self.error.emit(str(exc))


class _AnalysisWorker(QThread):
    """Runs the section analysis off the main thread."""

    progress = pyqtSignal(int)
    finished = pyqtSignal(object)  # AnalysisResult
    error = pyqtSignal(str)

    def __init__(self, bodies: list, spacing_mm: float, parent=None):
        super().__init__(parent)
        self._bodies = bodies       # list of (trimesh.Trimesh, density_kg_per_m3)
        self._spacing_mm = spacing_mm

    def run(self) -> None:
        try:
            from src.analysis.properties import run_full_analysis
            result = run_full_analysis(
                self._bodies,
                self._spacing_mm,
                progress_callback=lambda p: self.progress.emit(p),
            )
            self.finished.emit(result)
        except Exception as exc:
            self.error.emit(str(exc))


class MainWindow(QMainWindow):
    """Top-level application window."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("3D File Analyzer")
        self.resize(1400, 900)

        self._bodies: list = []              # list of trimesh.Trimesh, one per body
        self._density_spins: list[QDoubleSpinBox] = []
        self._load_worker: _LoadWorker | None = None
        self._analysis_worker: _AnalysisWorker | None = None

        self._build_menu()
        self._build_ui()

    def _build_menu(self) -> None:
        menubar = self.menuBar()

        file_menu = menubar.addMenu("&File")
        open_action = QAction("&Open STEP File…", self)
        open_action.setShortcut("Ctrl+O")
        open_action.triggered.connect(self.open_file)
        file_menu.addAction(open_action)

        export_action = QAction("&Export CSV…", self)
        export_action.setShortcut("Ctrl+E")
        export_action.triggered.connect(self.export_csv)
        file_menu.addAction(export_action)

        file_menu.addSeparator()
        quit_action = QAction("&Quit", self)
        quit_action.setShortcut("Ctrl+Q")
        quit_action.triggered.connect(self.close)
        file_menu.addAction(quit_action)

        help_menu = menubar.addMenu("&Help")
        about_action = QAction("&About", self)
        about_action.triggered.connect(self._show_about)
        help_menu.addAction(about_action)

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(6, 6, 6, 6)
        main_layout.setSpacing(6)

        # ── Controls bar ────────────────────────────────────────────────
        self._controls_group = QGroupBox("Analysis Parameters")
        self._controls_layout = QHBoxLayout(self._controls_group)

        # Density spinners placeholder (rebuilt after each file load).
        self._densities_widget = QWidget()
        self._densities_layout = QHBoxLayout(self._densities_widget)
        self._densities_layout.setContentsMargins(0, 0, 0, 0)
        self._densities_layout.addWidget(QLabel("Open a file to set densities."))
        self._controls_layout.addWidget(self._densities_widget)

        self._controls_layout.addSpacing(16)
        self._controls_layout.addWidget(QLabel("Section spacing:"))
        self._spacing_spin = QDoubleSpinBox()
        self._spacing_spin.setRange(0.001, 100000.0)
        self._spacing_spin.setValue(10.0)
        self._spacing_spin.setDecimals(3)
        self._spacing_spin.setSuffix(" mm")
        self._spacing_spin.setFixedWidth(130)
        self._controls_layout.addWidget(self._spacing_spin)

        self._controls_layout.addSpacing(16)
        self._load_btn = QPushButton("Open File…")
        self._load_btn.clicked.connect(self.open_file)
        self._controls_layout.addWidget(self._load_btn)

        self._analyze_btn = QPushButton("Analyze")
        self._analyze_btn.setEnabled(False)
        self._analyze_btn.clicked.connect(self.run_analysis)
        self._controls_layout.addWidget(self._analyze_btn)

        self._export_btn = QPushButton("Export CSV…")
        self._export_btn.setEnabled(False)
        self._export_btn.clicked.connect(self.export_csv)
        self._controls_layout.addWidget(self._export_btn)

        self._controls_layout.addStretch()

        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 100)
        self._progress_bar.setVisible(False)
        self._progress_bar.setFixedWidth(180)
        self._controls_layout.addWidget(self._progress_bar)

        main_layout.addWidget(self._controls_group)

        # ── Splitter: viewer + results ───────────────────────────────────
        splitter = QSplitter(Qt.Horizontal)

        self._viewer = ViewerPanel()
        self._viewer.setMinimumWidth(400)
        splitter.addWidget(self._viewer)

        self._results = ResultsPanel()
        self._results.setMinimumWidth(340)
        self._results.row_selected.connect(self._on_row_selected)
        splitter.addWidget(self._results)

        splitter.setSizes([820, 480])
        main_layout.addWidget(splitter)

        # ── Status bar ───────────────────────────────────────────────────
        self._status = QStatusBar()
        self.setStatusBar(self._status)
        self._status.showMessage("Ready — open a STEP file to begin.")

    def _rebuild_density_spinners(self, n_bodies: int) -> None:
        """Replace the densities widget contents with one spinner per body."""
        # Clear old widgets.
        while self._densities_layout.count():
            item = self._densities_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        self._density_spins.clear()

        for i in range(n_bodies):
            color = _BODY_COLORS[i % len(_BODY_COLORS)]
            label = QLabel(f"Body {i + 1} ({color}) density:")
            # Style the label text in the body's color so it matches the viewer.
            label.setStyleSheet(f"color: {color};")
            self._densities_layout.addWidget(label)

            spin = QDoubleSpinBox()
            spin.setRange(0.001, 30000.0)
            spin.setValue(_DEFAULT_DENSITIES[i] if i < len(_DEFAULT_DENSITIES) else 1000.0)
            spin.setDecimals(3)
            spin.setSuffix(" kg/m³")
            spin.setFixedWidth(155)
            self._densities_layout.addWidget(spin)
            self._density_spins.append(spin)

            if i < n_bodies - 1:
                self._densities_layout.addSpacing(10)

    # ── File operations ─────────────────────────────────────────────────

    def open_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open STEP File",
            "",
            "STEP Files (*.stp *.step);;All Files (*)",
        )
        if not path:
            return

        self._set_busy(True, f"Loading {os.path.basename(path)}…")
        self._results.clear()
        self._analyze_btn.setEnabled(False)
        self._export_btn.setEnabled(False)
        self._bodies.clear()

        self._load_worker = _LoadWorker(path, parent=self)
        self._load_worker.finished.connect(self._on_load_finished)
        self._load_worker.error.connect(self._on_worker_error)
        self._load_worker.start()

    def _on_load_finished(self, tm_list: list, pv_list: list) -> None:
        self._bodies = tm_list
        self._viewer.display_bodies(pv_list)
        self._rebuild_density_spinners(len(tm_list))

        watertight = all(m.is_watertight for m in tm_list)
        wt_note = "" if watertight else " (non-watertight — approximation mode)"
        total_verts = sum(len(m.vertices) for m in tm_list)
        total_faces = sum(len(m.faces) for m in tm_list)

        self._status.showMessage(
            f"Loaded {len(tm_list)} body/bodies — "
            f"{total_verts:,} vertices, {total_faces:,} faces{wt_note}"
        )
        self._analyze_btn.setEnabled(True)
        self._set_busy(False)

    # ── Analysis ────────────────────────────────────────────────────────

    def run_analysis(self) -> None:
        if not self._bodies:
            return

        spacing = self._spacing_spin.value()
        densities = [spin.value() for spin in self._density_spins]

        # Validate spacing against combined Z range.
        z_min = min(float(m.bounds[0, 2]) for m in self._bodies)
        z_max = max(float(m.bounds[1, 2]) for m in self._bodies)
        z_range = z_max - z_min

        if spacing >= z_range:
            QMessageBox.warning(
                self,
                "Invalid Spacing",
                f"Section spacing ({spacing:.3f} mm) must be smaller than the "
                f"model height ({z_range:.3f} mm).",
            )
            return

        self._set_busy(True, "Analyzing…")
        self._progress_bar.setVisible(True)
        self._progress_bar.setValue(0)

        bodies = list(zip(self._bodies, densities))

        self._analysis_worker = _AnalysisWorker(bodies, spacing, parent=self)
        self._analysis_worker.progress.connect(self._progress_bar.setValue)
        self._analysis_worker.finished.connect(self._on_analysis_finished)
        self._analysis_worker.error.connect(self._on_worker_error)
        self._analysis_worker.start()

    def _on_analysis_finished(self, result) -> None:
        self._progress_bar.setVisible(False)
        self._results.populate(result)
        self._viewer.draw_section_planes(result)
        self._export_btn.setEnabled(True)
        self._set_busy(False)
        self._status.showMessage(
            f"Analysis complete — {len(result.slices)} sections, "
            f"total mass {result.total_mass:.4f} kg, "
            f"CoG Z={result.center_of_gravity[2]:.3f} mm"
        )

    # ── Export ──────────────────────────────────────────────────────────

    def export_csv(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Results", "analysis_results.csv", "CSV Files (*.csv)"
        )
        if not path:
            return
        try:
            self._results.export_csv(path)
            self._status.showMessage(f"Exported to {path}")
        except Exception as exc:
            QMessageBox.critical(self, "Export Error", str(exc))

    # ── Helpers ─────────────────────────────────────────────────────────

    def _on_row_selected(self, row: int) -> None:
        if self._results._result is None:
            return
        slices = self._results._result.slices
        if 0 <= row < len(slices):
            self._viewer.highlight_slice(slices[row])

    def _on_worker_error(self, message: str) -> None:
        self._set_busy(False)
        self._progress_bar.setVisible(False)
        QMessageBox.critical(self, "Error", message)
        self._status.showMessage(f"Error: {message}")

    def _set_busy(self, busy: bool, message: str = "") -> None:
        self._load_btn.setEnabled(not busy)
        self._analyze_btn.setEnabled(not busy and bool(self._bodies))
        if message:
            self._status.showMessage(message)

    def _show_about(self) -> None:
        QMessageBox.about(
            self,
            "About 3D File Analyzer",
            "<b>3D File Analyzer</b><br><br>"
            "Loads multi-body STEP files, displays each body in a distinct color, "
            "and computes per-section engineering properties (volume, mass per body, "
            "combined area moment of inertia) plus the overall center of gravity "
            "and inertia tensor.<br><br>"
            "Units: mm · kg/m³ · kg · mm⁴",
        )

    def closeEvent(self, event) -> None:
        self._viewer.close_plotter()
        super().closeEvent(event)
