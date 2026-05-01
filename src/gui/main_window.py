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

from .viewer_panel import ViewerPanel
from .results_panel import ResultsPanel


class _LoadWorker(QThread):
    """Loads and tessellates a STEP file off the main thread."""

    finished = pyqtSignal(object, object)  # (trimesh.Trimesh, pyvista.PolyData)
    error = pyqtSignal(str)

    def __init__(self, filepath: str, parent=None):
        super().__init__(parent)
        self._filepath = filepath

    def run(self) -> None:
        try:
            from src.io.step_reader import load_step
            tm, pv = load_step(self._filepath)
            self.finished.emit(tm, pv)
        except Exception as exc:
            self.error.emit(str(exc))


class _AnalysisWorker(QThread):
    """Runs the section analysis off the main thread."""

    progress = pyqtSignal(int)
    finished = pyqtSignal(object)  # AnalysisResult
    error = pyqtSignal(str)

    def __init__(self, mesh, spacing_mm: float, density: float, parent=None):
        super().__init__(parent)
        self._mesh = mesh
        self._spacing_mm = spacing_mm
        self._density = density

    def run(self) -> None:
        try:
            from src.analysis.properties import run_full_analysis
            result = run_full_analysis(
                self._mesh,
                self._spacing_mm,
                self._density,
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

        self._trimesh = None
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
        controls_group = QGroupBox("Analysis Parameters")
        controls_layout = QHBoxLayout(controls_group)

        controls_layout.addWidget(QLabel("Density (kg/m³):"))
        self._density_spin = QDoubleSpinBox()
        self._density_spin.setRange(0.001, 30000.0)
        self._density_spin.setValue(7800.0)
        self._density_spin.setDecimals(3)
        self._density_spin.setSuffix(" kg/m³")
        self._density_spin.setFixedWidth(160)
        controls_layout.addWidget(self._density_spin)

        controls_layout.addSpacing(20)
        controls_layout.addWidget(QLabel("Section spacing (mm):"))
        self._spacing_spin = QDoubleSpinBox()
        self._spacing_spin.setRange(0.001, 100000.0)
        self._spacing_spin.setValue(10.0)
        self._spacing_spin.setDecimals(3)
        self._spacing_spin.setSuffix(" mm")
        self._spacing_spin.setFixedWidth(140)
        controls_layout.addWidget(self._spacing_spin)

        controls_layout.addSpacing(20)
        self._load_btn = QPushButton("Open File…")
        self._load_btn.clicked.connect(self.open_file)
        controls_layout.addWidget(self._load_btn)

        self._analyze_btn = QPushButton("Analyze")
        self._analyze_btn.setEnabled(False)
        self._analyze_btn.clicked.connect(self.run_analysis)
        controls_layout.addWidget(self._analyze_btn)

        self._export_btn = QPushButton("Export CSV…")
        self._export_btn.setEnabled(False)
        self._export_btn.clicked.connect(self.export_csv)
        controls_layout.addWidget(self._export_btn)

        controls_layout.addStretch()

        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 100)
        self._progress_bar.setVisible(False)
        self._progress_bar.setFixedWidth(180)
        controls_layout.addWidget(self._progress_bar)

        main_layout.addWidget(controls_group)

        # ── Splitter: viewer + results ───────────────────────────────────
        splitter = QSplitter(Qt.Horizontal)

        self._viewer = ViewerPanel()
        self._viewer.setMinimumWidth(400)
        splitter.addWidget(self._viewer)

        self._results = ResultsPanel()
        self._results.setMinimumWidth(340)
        self._results.row_selected.connect(self._on_row_selected)
        splitter.addWidget(self._results)

        splitter.setSizes([800, 500])
        main_layout.addWidget(splitter)

        # ── Status bar ───────────────────────────────────────────────────
        self._status = QStatusBar()
        self.setStatusBar(self._status)
        self._status.showMessage("Ready — open a STEP file to begin.")

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
        self._trimesh = None

        self._load_worker = _LoadWorker(path, parent=self)
        self._load_worker.finished.connect(self._on_load_finished)
        self._load_worker.error.connect(self._on_worker_error)
        self._load_worker.start()

    def _on_load_finished(self, tm, pv_mesh) -> None:
        self._trimesh = tm
        self._viewer.display_mesh(pv_mesh)

        watertight_note = "" if tm.is_watertight else " (non-watertight — approximation mode)"
        self._status.showMessage(
            f"Loaded: {len(tm.vertices):,} vertices, {len(tm.faces):,} faces{watertight_note}"
        )
        self._analyze_btn.setEnabled(True)
        self._set_busy(False)

    # ── Analysis ────────────────────────────────────────────────────────

    def run_analysis(self) -> None:
        if self._trimesh is None:
            return

        spacing = self._spacing_spin.value()
        density = self._density_spin.value()

        z_range = float(self._trimesh.bounds[1, 2] - self._trimesh.bounds[0, 2])
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

        self._analysis_worker = _AnalysisWorker(
            self._trimesh, spacing, density, parent=self
        )
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
            f"total mass {result.total_mass:.4f} kg"
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
        self._analyze_btn.setEnabled(not busy and self._trimesh is not None)
        if message:
            self._status.showMessage(message)

    def _show_about(self) -> None:
        QMessageBox.about(
            self,
            "About 3D File Analyzer",
            "<b>3D File Analyzer</b><br><br>"
            "Loads STEP files, displays them in 3D, and computes per-section "
            "engineering properties (volume, mass, area moment of inertia) "
            "plus the overall center of gravity and inertia tensor.<br><br>"
            "Units: mm · kg/m³ · kg · mm⁴",
        )

    def closeEvent(self, event) -> None:
        self._viewer.close_plotter()
        super().closeEvent(event)
