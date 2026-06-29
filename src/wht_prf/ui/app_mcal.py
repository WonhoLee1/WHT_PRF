# WHTOOLs MATCALIB 2026
"""
MCalibration-Style Material Calibration Dashboard.
3-Column Layout: Calibration Setup | Plot Area | Optimization
"""
import os, sys, json, logging
import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QPushButton, QLabel, QSpinBox, QComboBox, QProgressBar,
    QMessageBox, QFileDialog, QSplitter, QTabWidget, QToolBar,
    QToolButton, QMenu, QSizePolicy, QFrame, QDialog, QGroupBox,
    QTableWidget, QTableWidgetItem, QHeaderView, QCheckBox,
    QDoubleSpinBox, QSlider, QStackedWidget, QTextEdit, QLineEdit, QScrollArea
)
from PySide6.QtCore import Qt, QSize, QTimer, Signal
from PySide6.QtGui import QAction, QFont, QIcon

from wht_prf.ui.widgets.plot_canvas import PlotCanvas
from wht_prf.ui.widgets.case_manager import CaseManagerWidget
from wht_prf.ui.widgets.scenario_dialog import ScenarioCalibrationDialog
from wht_prf.ui.widgets.plot_manager_window import PlotManagerWindow
from wht_prf.ui.widgets.history_manager import HistoryManagerWidget
from wht_prf.ui.widgets.virtual_testing import VirtualTestingWidget
from wht_prf.ui.widgets.sensitivity_chart import SensitivityChartWidget
from wht_prf.ui.widgets.export_panel import ExportPanelWidget
from wht_prf.ui.widgets.material_library import MaterialLibraryDialog
from wht_prf.ui.widgets.report_generator import ReportGenerator
from wht_prf.ui.widgets.fitness_convergence import FitnessConvergenceWidget
from wht_prf.ui.widgets.model_selection_dialog import ModelSelectionDialog
from wht_prf.ui.widgets.parameter_tree_widget import ParameterTreeWidget
from wht_prf.ui.simulation import simulate_prf_steps_jit, simulate_prony_steps_jit
from wht_prf.ui.workers import CalibrationWorker, MCMCWorker
from wht_prf.core.fitness import METRIC_REGISTRY, METRIC_ORDER, get_metric

# ── MCalibration Dark Theme QSS ──────────────────────────────────────────────
STYLE_QSS = """
/* Global */
QMainWindow, QWidget {
    background-color: #1e1e1e;
    color: #d4d4d4;
    font-family: 'Segoe UI', Arial, sans-serif;
    font-size: 13px;
}

/* Tabs */
QTabWidget::pane {
    border: 1px solid #3c3c3c;
    background: #252526;
}
QTabBar::tab {
    background: #2d2d2d;
    border: 1px solid #3c3c3c;
    padding: 8px 16px;
    margin-right: 2px;
    color: #858585;
}
QTabBar::tab:selected {
    background: #1e1e1e;
    color: #ffffff;
    border-bottom: 2px solid #007acc;
}
QTabBar::tab:hover:!selected {
    background: #3c3c3c;
}

/* Tables */
QTableWidget {
    background: #1e1e1e;
    alternate-background-color: #252526;
    border: 1px solid #3c3c3c;
    gridline-color: #3c3c3c;
    color: #d4d4d4;
    font-size: 13px;
    selection-background-color: #264f78;
}
QTableWidget::item {
    padding: 4px 8px;
}
QTableWidget::item:selected {
    background: #264f78;
}
QHeaderView::section {
    background: #2d2d2d;
    color: #cccccc;
    border: none;
    border-right: 1px solid #3c3c3c;
    border-bottom: 1px solid #3c3c3c;
    padding: 6px 8px;
    font-weight: bold;
    font-size: 13px;
}

/* Buttons */
QPushButton {
    background: #0e639c;
    border: none;
    border-radius: 4px;
    color: #ffffff;
    padding: 8px 16px;
    font-weight: bold;
    min-height: 20px;
}
QPushButton:hover {
    background: #1177bb;
}
QPushButton:pressed {
    background: #094771;
}
QPushButton:disabled {
    background: #3c3c3c;
    color: #6c6c6c;
}
QPushButton.secondary {
    background: #3c3c3c;
    border: 1px solid #555555;
}
QPushButton.secondary:hover {
    background: #505050;
}

/* ComboBox */
QComboBox {
    background: #3c3c3c;
    border: 1px solid #555555;
    border-radius: 4px;
    color: #d4d4d4;
    padding: 6px 10px;
    min-height: 20px;
}
QComboBox:hover {
    border-color: #007acc;
}
QComboBox::drop-down {
    border: none;
    width: 20px;
}
QComboBox QAbstractItemView {
    background: #2d2d2d;
    color: #d4d4d4;
    selection-background-color: #264f78;
    border: 1px solid #3c3c3c;
}

/* SpinBox */
QSpinBox, QDoubleSpinBox {
    background: #3c3c3c;
    border: 1px solid #555555;
    border-radius: 4px;
    color: #d4d4d4;
    padding: 4px 8px;
}
QSpinBox:hover, QDoubleSpinBox:hover {
    border-color: #007acc;
}

/* Progress Bar */
QProgressBar {
    background: #3c3c3c;
    border: none;
    border-radius: 4px;
    text-align: center;
    color: #ffffff;
    height: 20px;
}
QProgressBar::chunk {
    background: #007acc;
    border-radius: 4px;
}

/* Splitter */
QSplitter::handle {
    background: #3c3c3c;
    width: 2px;
    height: 2px;
}
QSplitter::handle:hover {
    background: #007acc;
}

/* Group Box */
QGroupBox {
    border: 1px solid #3c3c3c;
    border-radius: 4px;
    margin-top: 12px;
    padding-top: 16px;
    font-weight: bold;
    color: #cccccc;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 12px;
    padding: 0 6px;
    background: #252526;
}

/* Scroll Bar */
QScrollBar:vertical {
    background: #1e1e1e;
    width: 12px;
    border-radius: 6px;
}
QScrollBar::handle:vertical {
    background: #424242;
    border-radius: 6px;
    min-height: 30px;
}
QScrollBar::handle:vertical:hover {
    background: #555555;
}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0;
}

/* Check Box */
QCheckBox {
    spacing: 8px;
    color: #d4d4d4;
}
QCheckBox::indicator {
    width: 18px;
    height: 18px;
    border: 2px solid #555555;
    border-radius: 4px;
    background: #3c3c3c;
}
QCheckBox::indicator:checked {
    background: #007acc;
    border-color: #007acc;
}

/* Slider */
QSlider::groove:horizontal {
    background: #3c3c3c;
    height: 6px;
    border-radius: 3px;
}
QSlider::handle:horizontal {
    background: #007acc;
    width: 14px;
    height: 14px;
    margin: -4px 0;
    border-radius: 7px;
}
QSlider::sub-page:horizontal {
    background: #007acc;
    border-radius: 3px;
}

/* Tool Bar */
QToolBar {
    background: #2d2d2d;
    border-bottom: 1px solid #3c3c3c;
    spacing: 8px;
    padding: 6px;
}
QToolButton {
    background: transparent;
    border: 1px solid transparent;
    border-radius: 4px;
    color: #d4d4d4;
    padding: 6px 12px;
    min-width: 60px;
}
QToolButton:hover {
    background: #3c3c3c;
    border-color: #555555;
}
QToolButton:pressed, QToolButton:checked {
    background: #094771;
    border-color: #007acc;
}

/* Labels */
QLabel {
    color: #d4d4d4;
}
QLabel.section-header {
    font-weight: bold;
    font-size: 13px;
    color: #ffffff;
    padding: 8px 0;
}
QLabel.badge {
    background: #007acc;
    color: #ffffff;
    padding: 2px 8px;
    border-radius: 10px;
    font-size: 13px;
    font-weight: bold;
}
"""


# ── Collapsible Section Widget ────────────────────────────────────────────────
class CollapsibleSection(QWidget):
    """Collapsible section with arrow indicator and checkbox."""
    toggled = Signal(bool)
    
    def __init__(self, title: str, badge: str = "", parent=None):
        super().__init__(parent)
        self._is_expanded = True
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # Header row
        header = QHBoxLayout()
        header.setSpacing(8)
        
        self._arrow = QLabel("▼")
        self._arrow.setFixedWidth(16)
        self._arrow.setStyleSheet("font-size: 13px; color: #858585;")
        header.addWidget(self._arrow)
        
        self._checkbox = QCheckBox()
        self._checkbox.setChecked(True)
        self._checkbox.stateChanged.connect(self._on_checkbox_changed)
        header.addWidget(self._checkbox)
        
        self._title_label = QLabel(title)
        self._title_label.setStyleSheet("font-weight: bold; font-size: 13px; color: #ffffff;")
        header.addWidget(self._title_label)
        
        if badge:
            badge_label = QLabel(badge)
            badge_label.setProperty("class", "badge")
            badge_label.setStyleSheet(
                "background: #007acc; color: #ffffff; padding: 2px 8px; "
                "border-radius: 10px; font-size: 13px; font-weight: bold;"
            )
            header.addWidget(badge_label)
        
        header.addStretch()
        
        self._header_widget = QWidget()
        self._header_widget.setLayout(header)
        self._header_widget.setStyleSheet("QWidget:hover { background: #2a2d2e; }")
        self._header_widget.setCursor(Qt.PointingHandCursor)
        self._header_widget.mousePressEvent = self._toggle
        
        layout.addWidget(self._header_widget)
        
        # Content area
        self._content = QWidget()
        self._content_layout = QVBoxLayout(self._content)
        self._content_layout.setContentsMargins(24, 8, 8, 8)
        self._content_layout.setSpacing(4)
        layout.addWidget(self._content)
    
    def content_layout(self):
        return self._content_layout
    
    def _toggle(self, event=None):
        self._is_expanded = not self._is_expanded
        self._content.setVisible(self._is_expanded)
        self._arrow.setText("▼" if self._is_expanded else "▶")
    
    def _on_checkbox_changed(self, state):
        self.toggled.emit(state == Qt.Checked)
    
    def is_checked(self):
        return self._checkbox.isChecked()


# ── Sensitivity Bar Widget ──────────────────────────────────────────────────
class SensitivityBar(QWidget):
    """Colored sensitivity indicator bar."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedWidth(60)
        self._value = 0.0
    
    def set_value(self, value: float):
        self._value = max(0.0, min(1.0, value))
        self.update()
    
    def paintEvent(self, event):
        from PySide6.QtGui import QPainter, QColor
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Background
        painter.setBrush(QColor("#3c3c3c"))
        painter.setPen(Qt.NoPen)
        painter.drawRoundedRect(0, 2, self.width(), self.height() - 4, 3, 3)
        
        # Value bar
        if self._value > 0:
            bar_width = int(self.width() * self._value)
            if self._value > 0.7:
                color = QColor("#4caf50")  # Green
            elif self._value > 0.3:
                color = QColor("#ff9800")  # Orange
            else:
                color = QColor("#f44336")  # Red
            
            painter.setBrush(color)
            painter.drawRoundedRect(0, 2, bar_width, self.height() - 4, 3, 3)
        
        painter.end()


# ── Parameter Editor with Sensitivity ──────────────────────────────────────
class MCalParameterEditor(QWidget):
    """Parameter editor with Tree-Table structure (Hyperelastic + Viscoelastic unified)."""
    parameter_changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._he_type = "YEOH"
        self._num_networks = 1
        self._creep_law = "TIME"
        self._params = {}
        self._build_ui()

    def _build_ui(self):
        """Tree-Table parameter editor with Hyperelastic + Viscoelastic unified."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)

        # Title
        title = QLabel("Material Parameters")
        title.setProperty("class", "section-header")
        layout.addWidget(title)

        # Scroll area for tree widget
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(200)

        self._tree = ParameterTreeWidget(self)
        self._tree.parameter_changed.connect(self._on_param_changed)
        scroll.setWidget(self._tree)

        layout.addWidget(scroll)

    def _on_param_changed(self):
        """Forward parameter change signal."""
        self.parameter_changed.emit()

    def get_parameters(self) -> dict:
        """Return current parameters from tree widget."""
        return self._tree.get_parameters()

    def set_parameters(self, params: dict):
        """Load parameters into tree widget."""
        self._params = params
        self._he_type = params.get("hyperelastic_type", "YEOH")
        self._tree.set_parameters(params)

    def set_model_config(self, he_type: str, num_networks: int, creep_law: str):
        """Update model configuration."""
        self._he_type = he_type
        self._num_networks = num_networks
        self._creep_law = creep_law

    def set_model_mode(self, mode: str):
        """Set model mode (PRF/Prony)."""
        pass


# ── Test Datasets Table ──────────────────────────────────────────────────
class TestDatasetsWidget(QWidget):
    """Test datasets table matching MCalibration style."""
    dataset_changed = Signal()
    
    def __init__(self, case_manager, parent=None):
        super().__init__(parent)
        self._case_manager = case_manager
        self._build_ui()
    
    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        
        # Header
        header = QLabel("Test datasets")
        header.setProperty("class", "section-header")
        layout.addWidget(header)
        
        # Table
        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["Name", "Wt", "R²"])
        # 열폭 조절 가능하게 설정
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        self.table.horizontalHeader().setStretchLastSection(False)
        self.table.setColumnWidth(0, 100)
        self.table.setColumnWidth(1, 60)
        self.table.setColumnWidth(2, 70)
        # 스크롤 지원
        self.table.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.table.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QTableWidget.DoubleClicked)
        self.table.cellChanged.connect(self._on_cell_changed)
        
        layout.addWidget(self.table)
    
    def refresh(self):
        self.table.blockSignals(True)
        self.table.setRowCount(0)
        
        for ds in self._case_manager.get_active_datasets():
            row = self.table.rowCount()
            self.table.insertRow(row)
            
            # Name
            name_item = QTableWidgetItem(ds.get("name", "Unknown"))
            name_item.setFlags(Qt.ItemIsEnabled)
            self.table.setItem(row, 0, name_item)
            
            # Weight
            weight_item = QTableWidgetItem("1.0")
            self.table.setItem(row, 1, weight_item)
            
            # R²
            r2 = ds.get("r2", None)
            if r2 is not None:
                r2_item = QTableWidgetItem(f"{r2:.3f}")
                if r2 >= 0.99:
                    r2_item.setForeground(QColor("#4caf50"))
                elif r2 >= 0.95:
                    r2_item.setForeground(QColor("#ff9800"))
                else:
                    r2_item.setForeground(QColor("#f44336"))
            else:
                r2_item = QTableWidgetItem("—")
            r2_item.setFlags(Qt.ItemIsEnabled)
            self.table.setItem(row, 2, r2_item)
        
        self.table.blockSignals(False)
    
    def _on_cell_changed(self, row, col):
        if col == 1:  # Weight column
            self.dataset_changed.emit()


# ── Optimization Settings Panel ────────────────────────────────────────────
class OptimizationPanel(QWidget):
    """Right panel with optimization settings."""
    run_calibration = Signal()
    stop_calibration = Signal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._build_ui()
    
    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        
        # Header
        header = QLabel("Optimization")
        header.setProperty("class", "section-header")
        layout.addWidget(header)
        
        # Algorithm
        algo_layout = QHBoxLayout()
        algo_layout.addWidget(QLabel("Algorithm"))
        self.algo_combo = QComboBox()
        self.algo_combo.addItems(["Particle Swarm", "Genetic Algorithm", "Gradient Descent"])
        algo_layout.addWidget(self.algo_combo)
        layout.addLayout(algo_layout)
        
        # Error measure
        error_layout = QHBoxLayout()
        error_layout.addWidget(QLabel("Error measure"))
        self.error_combo = QComboBox()
        self.error_combo.addItems(["R² (coeff. det.)", "RMSE", "MAE", "NMAD"])
        error_layout.addWidget(self.error_combo)
        layout.addLayout(error_layout)
        
        # Settings grid
        settings_layout = QHBoxLayout()
        
        # Tolerance
        tol_layout = QVBoxLayout()
        tol_layout.addWidget(QLabel("Tolerance"))
        self.tol_spin = QDoubleSpinBox()
        self.tol_spin.setRange(1e-10, 1e-1)
        self.tol_spin.setDecimals(10)
        self.tol_spin.setSingleStep(1e-6)
        self.tol_spin.setValue(1e-6)
        tol_layout.addWidget(self.tol_spin)
        settings_layout.addLayout(tol_layout)
        
        # CPUs
        cpu_layout = QVBoxLayout()
        cpu_layout.addWidget(QLabel("CPUs"))
        self.cpu_spin = QSpinBox()
        self.cpu_spin.setRange(1, 64)
        self.cpu_spin.setValue(8)
        cpu_layout.addWidget(self.cpu_spin)
        settings_layout.addLayout(cpu_layout)
        
        layout.addLayout(settings_layout)
        
        # Max evaluations
        maxeval_layout = QVBoxLayout()
        maxeval_layout.addWidget(QLabel("Max evaluations"))
        self.maxeval_spin = QSpinBox()
        self.maxeval_spin.setRange(100, 1000000)
        self.maxeval_spin.setSingleStep(1000)
        self.maxeval_spin.setValue(10000)
        maxeval_layout.addWidget(self.maxeval_spin)
        layout.addLayout(maxeval_layout)
        
        # Run button
        self.run_btn = QPushButton("▶ Run calibration")
        self.run_btn.setMinimumHeight(40)
        self.run_btn.setStyleSheet(
            "QPushButton { background: #0e639c; font-size: 14px; font-weight: bold; }"
            "QPushButton:hover { background: #1177bb; }"
        )
        self.run_btn.clicked.connect(self.run_calibration.emit)
        layout.addWidget(self.run_btn)
        
        # Objective history
        obj_header = QLabel("Objective history")
        obj_header.setStyleSheet("font-size: 13px; color: #858585; margin-top: 8px;")
        layout.addWidget(obj_header)
        
        self.obj_plot = PlotCanvas(width=3, height=1.5, dpi=90)
        self.obj_plot.toolbar.hide()
        layout.addWidget(self.obj_plot)
        
        # Stats
        stats_layout = QHBoxLayout()
        self.obj_label = QLabel("Objective: —")
        self.obj_label.setStyleSheet("font-size: 13px; color: #4caf50;")
        stats_layout.addWidget(self.obj_label)
        layout.addLayout(stats_layout)
        
        eval_layout = QHBoxLayout()
        self.eval_label = QLabel("Evaluations: —")
        self.eval_label.setStyleSheet("font-size: 13px; color: #858585;")
        eval_layout.addWidget(self.eval_label)
        self.time_label = QLabel("Elapsed: —")
        self.time_label.setStyleSheet("font-size: 13px; color: #858585;")
        eval_layout.addWidget(self.time_label)
        layout.addLayout(eval_layout)
        
        # Status
        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet("font-size: 13px; color: #4caf50; font-weight: bold;")
        layout.addWidget(self.status_label)
        
        layout.addStretch()
    
    def update_objective(self, value: float, evals: int, elapsed: float):
        self.obj_label.setText(f"Objective: {value:.5f}")
        self.eval_label.setText(f"Evaluations: {evals:,}")
        self.time_label.setText(f"Elapsed: {elapsed:.1f} s")
        
        if value >= 0.99:
            self.status_label.setText("Converged · R²")
            self.status_label.setStyleSheet("font-size: 13px; color: #4caf50; font-weight: bold;")
        elif value >= 0.95:
            self.status_label.setText("Improving · R²")
            self.status_label.setStyleSheet("font-size: 13px; color: #ff9800; font-weight: bold;")
        else:
            self.status_label.setText("Running...")
            self.status_label.setStyleSheet("font-size: 13px; color: #d4d4d4;")
    
    def add_objective_point(self, x: float, y: float):
        self.obj_plot.ax.plot(x, y, 'o', color='#4caf50', markersize=4)
        self.obj_plot.draw_data()


# ── Main Application ──────────────────────────────────────────────────────
class MCalibrationApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("WHTOOLs MatCalib 2026")
        self.resize(QSize(1500, 920))
        
        # Widgets
        self.case_manager = CaseManagerWidget()
        self.param_editor = MCalParameterEditor()
        self.preview_canvas = PlotCanvas(width=8, height=5, dpi=100)
        self.cal_canvas = PlotCanvas(width=8, height=5, dpi=100)
        self.history_manager = HistoryManagerWidget()
        self.history_manager.load_history_params.connect(self._on_history_load)
        self.history_manager.history_selected.connect(self.on_history_compare_selected)
        self.sensitivity_widget = SensitivityChartWidget(
            get_current_params_fn=self.param_editor.get_parameters,
            get_active_datasets_fn=self.case_manager.get_active_datasets
        )
        # recommend_panel removed (moved to ModelSelectionDialog)
        self.material_library = MaterialLibraryDialog()
        self.material_library.get_current_params_callback = self.param_editor.get_parameters
        self.export_panel = ExportPanelWidget(get_current_params_fn=self.param_editor.get_parameters)
        self.virtual_testing = VirtualTestingWidget(get_current_params_fn=self.param_editor.get_parameters)
        self.fitness_convergence = FitnessConvergenceWidget()
        self.optimization_panel = OptimizationPanel()
        self.test_datasets = TestDatasetsWidget(self.case_manager)
        
        self.setup_ribbon()
        self.setup_central_layout()
        self.setup_connections()
        self.init_default_material()
        
        self.cal_worker = None
        self.mcmc_worker = None

        # 메뉴바 설정
        self.setup_menu_bar()

    def setup_menu_bar(self):
        """메뉴바 설정"""
        menubar = self.menuBar()

        # 기능 메뉴
        menu_func = menubar.addMenu("기능")
        # 향후 추가될 메뉴 항목들을 위한 placeholder
        menu_func.addAction("데이터 전처리...")  # TODO: 향후 구현
        menu_func.addAction("원점 보정...")      # TODO: 향후 구현
        menu_func.addAction("다운샘플링...")     # TODO: 향후 구현

    def setup_ribbon(self):
        ribbon = QToolBar()
        ribbon.setMovable(False)
        ribbon.setIconSize(QSize(24, 24))
        self.addToolBar(ribbon)
        
        # File group
        self._add_ribbon_group(ribbon, "FILE", [
            ("💾 Save", self.save_session),
            ("📂 Load", self.load_session),
            ("📄 Report", self.generate_html_report_file),
        ])
        
        # Data group
        self._add_ribbon_group(ribbon, "DATA", [
            ("📊 Load", self.case_manager.on_load_files),
            ("📦 Library", lambda: self.material_library.exec()),
        ])
        
        # Model selection via popup (legacy controls removed)
        self.btn_model_select = QPushButton("⚙️ Select Model")
        self.btn_model_select.clicked.connect(self._open_model_selection)
        ribbon.addWidget(self.btn_model_select)
        ribbon.addSeparator()
        
        # Note: Epochs and Fitness controls removed from ribbon (moved to elsewhere or removed)
        
        # Tools group
        self._add_ribbon_group(ribbon, "TOOLS", [
            ("🧪 Virtual", lambda: self._open_dialog("Virtual Testing", self.virtual_testing)),
            ("📈 Plots", self.open_plot_manager),
            ("📖 Wizard", self.open_scenario_wizard),
        ])
    
    def _add_ribbon_group(self, ribbon, label, buttons):
        sep = QFrame()
        sep.setFrameShape(QFrame.VLine)
        sep.setStyleSheet("color: #3c3c3c;")
        ribbon.addWidget(sep)
        
        grp = QWidget()
        gl = QVBoxLayout(grp)
        gl.setContentsMargins(2, 0, 2, 0)
        gl.setSpacing(1)
        gl.addWidget(QLabel(label), alignment=Qt.AlignCenter)
        
        bl = QHBoxLayout()
        bl.setSpacing(2)
        for icon_text, handler in buttons:
            btn = QToolButton()
            btn.setText(icon_text)
            btn.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
            btn.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
            btn.setMinimumHeight(48)
            btn.clicked.connect(handler)
            bl.addWidget(btn)
        
        gl.addLayout(bl)
        ribbon.addWidget(grp)
    
    def setup_central_layout(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(4, 4, 4, 4)
        main_layout.setSpacing(4)
        
        # Material tabs
        self.material_tabs = QTabWidget()
        self.material_tabs.setTabsClosable(True)
        self.material_tabs.setMovable(True)
        self.material_tabs.addTab(self._create_material_tab(), "PP_PC3TF2")
        
        # Search bar
        search_layout = QHBoxLayout()
        search_layout.addStretch()
        search_input = QLineEdit()
        search_input.setPlaceholderText("🔍 Search materials...")
        search_input.setFixedWidth(200)
        search_layout.addWidget(search_input)
        
        top_bar = QHBoxLayout()
        top_bar.addWidget(self.material_tabs)
        top_bar.addLayout(search_layout)
        main_layout.addLayout(top_bar)
        
        # Main 3-column splitter
        h_splitter = QSplitter(Qt.Horizontal)
        
        # Left panel
        left_panel = self._create_left_panel()
        h_splitter.addWidget(left_panel)
        
        # Center panel
        center_panel = self._create_center_panel()
        h_splitter.addWidget(center_panel)
        
        # Right panel
        right_panel = self._create_right_panel()
        h_splitter.addWidget(right_panel)

        # 왼쪽 패널을 화면의 1/3로 설정 (500, 500, 500)
        h_splitter.setSizes([500, 500, 500])
        h_splitter.setStretchFactor(0, 1)  # Left panel 크기 조절 가능
        h_splitter.setStretchFactor(1, 1)  # Center panel 크기 조절 가능
        h_splitter.setStretchFactor(2, 1)  # Right panel 크기 조절 가능
        main_layout.addWidget(h_splitter, stretch=1)
        
        # Bottom bar
        bottom_bar = self._create_bottom_bar()
        main_layout.addWidget(bottom_bar)
    
    def _create_material_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.preview_canvas)
        return widget
    
    def _create_left_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        # Calibration setup header
        setup_header = QHBoxLayout()
        setup_label = QLabel("Calibration setup")
        setup_label.setProperty("class", "section-header")
        setup_header.addWidget(setup_label)
        setup_header.addStretch()

        layout.addLayout(setup_header)

        # Vertical splitter for resizable panels
        v_splitter = QSplitter(Qt.Vertical)

        # Case manager
        v_splitter.addWidget(self.case_manager)

        # Parameter editor
        v_splitter.addWidget(self.param_editor)

        # Test datasets
        v_splitter.addWidget(self.test_datasets)

        # Equal initial sizes (can be adjusted by user)
        v_splitter.setSizes([200, 300, 150])
        v_splitter.setCollapsible(0, False)
        v_splitter.setCollapsible(1, False)
        v_splitter.setCollapsible(2, False)

        layout.addWidget(v_splitter, stretch=1)

        # Note: recommend_panel removed (functionality moved to ModelSelectionDialog)

        return panel
    
    def _create_center_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)
        
        # Plot tabs
        self.plot_tabs = QTabWidget()
        self.plot_tabs.addTab(self.preview_canvas, "Stress relaxation")
        self.plot_tabs.addTab(self.cal_canvas, "ST stress-strain")
        self.plot_tabs.addTab(self.sensitivity_widget, "Stress-time")
        layout.addWidget(self.plot_tabs, stretch=1)
        
        # Auto-update checkbox
        auto_update = QCheckBox("Auto-update response plots")
        auto_update.setChecked(True)
        layout.addWidget(auto_update)
        
        return panel
    
    def _create_right_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.addWidget(self.optimization_panel)
        return panel
    
    def _create_bottom_bar(self) -> QWidget:
        bar = QWidget()
        bar.setFixedHeight(40)
        bar.setStyleSheet("background: #2d2d2d; border-top: 1px solid #3c3c3c;")
        
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(8, 4, 8, 4)
        
        tools_btn = QPushButton("⚙ Calibration tools")
        tools_btn.setFixedHeight(28)
        tools_btn.setStyleSheet("background: transparent; border: none; color: #858585;")
        layout.addWidget(tools_btn)
        
        export_btn = QPushButton("📦 Export Abaqus / Radioss deck")
        export_btn.setFixedHeight(28)
        export_btn.setStyleSheet("background: transparent; border: none; color: #858585;")
        export_btn.clicked.connect(lambda: self._open_dialog("Export", self.export_panel))
        layout.addWidget(export_btn)
        
        layout.addStretch()
        
        self.status_bar = QLabel("Ready")
        self.status_bar.setStyleSheet("color: #4caf50; font-weight: bold;")
        layout.addWidget(self.status_bar)
        
        return bar
    
    def setup_connections(self):
        self.case_manager.cases_updated.connect(self.on_cases_updated)
        self.param_editor.parameter_changed.connect(self.run_fast_preview)
        self.material_library.material_loaded.connect(self.on_material_loaded_from_lib)
        self.export_panel.material_imported.connect(self.on_material_imported_from_fea)
        self.optimization_panel.run_calibration.connect(self.run_calibration)
        self.optimization_panel.stop_calibration.connect(self.cancel_calibration)
    
    def init_default_material(self):
        default_mat = {
            "hyperelastic_type": "YEOH",
            "hyperelastic_params": [0.5, 0.0, 0.0],
            "networks": [{"stiffness_ratio": 0.3, "creep_params": [1.5e-5, 3.0, 0.0]}]
        }
        self.param_editor.set_parameters(default_mat)
    
    def on_model_config_changed(self):
        he_map = {
            "Yeoh": "YEOH", "Neo-Hookean": "NEO_HOOKEAN",
            "Arruda-Boyce": "ARRUDA_BOYCE", "Ogden": "OGDEN",
            "Van der Waals": "VAN_DER_WAALS"
        }
        creep_map = {
            "Power Law": "TIME", "Strain Hardening": "STRAIN",
            "Bergstrom-Boyce": "BERGSTROM", "Hyperbolic Sine": "HYPERB"
        }
        self.param_editor.set_model_config(
            he_map.get(self.cmb_he_type.currentText(), "YEOH"),
            self.spin_num_nets.value(),
            creep_map.get(self.cmb_creep_law.currentText(), "TIME")
        )
        self.run_fast_preview()

    def _open_model_selection(self):
        """모델 선택 다이얼로그 열기"""
        # 활성 데이터셋 수집
        active_datasets = self.case_manager.get_active_datasets()

        # 다이얼로그 생성 및 데이터셋 설정
        dlg = ModelSelectionDialog(self, active_datasets)
        dlg.set_datasets(active_datasets)

        if dlg.exec() == QDialog.Accepted:
            he_type, model_mode, num_networks = dlg.get_model_config()

            # 파라미터 에디터에 모델 설정 업데이트
            self.param_editor.set_model_config(he_type, num_networks, "TIME")

            # 메시지 박스로 선택 확인
            QMessageBox.information(
                self, "Model Selected",
                f"Selected: {he_type} + {model_mode}\nNetworks: {num_networks}"
            )

            # 파라미터 테이블 업데이트
            self.on_model_config_changed()

    def run_fast_preview(self):
        mat_data = self.param_editor.get_parameters()
        active_ds = self.case_manager.get_active_datasets()
        
        self.preview_canvas.clear()
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        
        for idx, ds in enumerate(active_ds):
            c = colors[idx % len(colors)]
            strain = ds["F_history"][:, 0, 0] - 1.0
            stress = ds["target_diff"]
            times = ds["times"]
            
            self.preview_canvas.ax.plot(times, stress, 'o', markersize=3, color=c, alpha=0.4, label=f"Exp: {ds['name']}")
            
            try:
                dts = jnp.diff(jnp.array(times), prepend=0.0)
                dts = dts.at[0].set(dts[1] if len(dts) > 1 else 1e-6)
                ctrl = jnp.zeros(len(times))
                inputs = jnp.array(1.0 + strain)
                targets = jnp.zeros(len(times))
                
                if self.param_editor._he_type == "PRONY":
                    _, _, stresses = simulate_prony_steps_jit(mat_data, dts, jnp.array(strain))
                else:
                    _, _, stresses = simulate_prf_steps_jit(mat_data, dts, ctrl, inputs, targets)
                
                self.preview_canvas.ax.plot(times, stresses, '-', color=c, linewidth=2, label=f"Sim: {ds['name']}")
            except Exception as e:
                logging.getLogger(__name__).warning("Preview failed: %s", e)
        
        if active_ds:
            self.preview_canvas.ax.set_title("Stress relaxation — test vs response")
            self.preview_canvas.ax.set_xlabel("Time (s)")
            self.preview_canvas.ax.set_ylabel("Stress (MPa)")
            self.preview_canvas.ax.legend(fontsize=8)
        
        self.preview_canvas.draw_data()
    
    def run_calibration(self):
        mat_data = self.param_editor.get_parameters()
        active_ds = self.case_manager.get_active_datasets()
        
        if not active_ds:
            QMessageBox.warning(self, "No Data", "Please activate at least 1 experimental dataset.")
            return
        
        self.optimization_panel.run_btn.setEnabled(False)
        self.optimization_panel.status_label.setText("Running...")
        
        datasets_backend = [
            {"times": jnp.array(d["times"]), "F_history": jnp.array(d["F_history"]),
             "target_diff": jnp.array(d["target_diff"])}
            for d in active_ds
        ]
        
        # Default fitness metric and epochs (removed from UI)
        metric_key = "mse"
        max_epochs = 150

        self.fitness_convergence.set_metric(metric_key)
        self.fitness_convergence.reset()

        self.cal_worker = CalibrationWorker(
            init_params=mat_data,
            datasets=datasets_backend,
            max_epochs=max_epochs,
            model_mode="PRF",
            metric_key=metric_key
        )
        self.cal_worker.epoch_progress.connect(self.on_calibration_progress)
        self.cal_worker.finished_success.connect(self.on_calibration_finished)
        self.cal_worker.per_case_fitness.connect(self.on_per_case_fitness)
        self.cal_worker.failed.connect(self.on_calibration_failed)
        self.cal_worker.start()
    
    def on_calibration_progress(self, epoch, loss, mat_params):
        self.fitness_convergence.on_loss_progress(epoch, loss, mat_params)
        self.optimization_panel.update_objective(loss, epoch + 1, 0)
    
    def on_per_case_fitness(self, results, metric_key):
        pass
    
    def on_calibration_finished(self, best_params, loss_history):
        self.param_editor.set_parameters(best_params)
        self.optimization_panel.run_btn.setEnabled(True)
        self.optimization_panel.status_label.setText("Converged · R²")
        self.optimization_panel.status_label.setStyleSheet("font-size: 13px; color: #4caf50; font-weight: bold;")
        self.test_datasets.refresh()
        self.run_fast_preview()
        QMessageBox.information(self, "Calibration Complete", "Optimization finished successfully.")
    
    def on_calibration_failed(self, err_msg):
        self.optimization_panel.run_btn.setEnabled(True)
        self.optimization_panel.status_label.setText("Failed")
        self.optimization_panel.status_label.setStyleSheet("font-size: 13px; color: #f44336; font-weight: bold;")
        QMessageBox.critical(self, "Calibration Failed", err_msg)
    
    def cancel_calibration(self):
        if self.cal_worker:
            self.cal_worker.stop()
    
    def on_cases_updated(self):
        # recommend_panel removed (moved to ModelSelectionDialog)
        self.test_datasets.refresh()
        self.run_fast_preview()
    
    def on_material_loaded_from_lib(self, mat_data):
        self.param_editor.set_parameters(mat_data)
        self.run_fast_preview()
    
    def on_material_imported_from_fea(self, mat_data):
        self.param_editor.set_parameters(mat_data)
        self.run_fast_preview()
    
    def on_history_compare_selected(self, selected_runs):
        self.cal_canvas.clear()
        active_ds = self.case_manager.get_active_datasets()
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        
        for idx, ds in enumerate(active_ds):
            c = colors[idx % len(colors)]
            self.cal_canvas.ax.plot(ds["times"], ds["target_diff"], 'o', markersize=3, color=c, alpha=0.3)
            
            for ri, run in enumerate(selected_runs):
                style = ['-', '--', ':', '-.'][ri % 4]
                try:
                    dts = jnp.diff(jnp.array(ds["times"]), prepend=0.0)
                    dts = dts.at[0].set(dts[1] if len(dts) > 1 else 1e-6)
                    if "prony_series" in run["params"]:
                        _, _, stresses = simulate_prony_steps_jit(run["params"], dts, jnp.array(ds["F_history"][:, 0, 0] - 1.0))
                    else:
                        _, _, stresses = simulate_prf_steps_jit(run["params"], dts, jnp.zeros(len(ds["times"])), jnp.array(ds["F_history"][:, 0, 0]), jnp.zeros(len(ds["times"])))
                    self.cal_canvas.ax.plot(ds["times"], stresses, style, color=c, linewidth=2, label=f"{run['name']}" if idx == 0 else "")
                except Exception as e:
                    logging.getLogger(__name__).warning("History compare failed: %s", e)
        
        self.cal_canvas.ax.set_title("Calibration History Comparison")
        self.cal_canvas.ax.set_xlabel("Time (s)")
        self.cal_canvas.ax.set_ylabel("Stress (MPa)")
        self.cal_canvas.ax.legend()
        self.cal_canvas.draw_data()
    
    def _on_history_load(self, params):
        self.param_editor.set_parameters(params)
    
    def _open_dialog(self, title, widget):
        dlg = QDialog(self)
        dlg.setWindowTitle(title)
        dlg.resize(1000, 700)
        lay = QVBoxLayout(dlg)
        lay.addWidget(widget)
        dlg.exec()
    
    def open_scenario_wizard(self):
        ScenarioCalibrationDialog(self).exec()
    
    def open_plot_manager(self):
        PlotManagerWindow(self).show()
    
    def save_session(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save Session", "", "JSON (*.json)")
        if not path:
            return
        mat_data = self.param_editor.get_parameters()
        cases_s = [
            {"name": c["name"], "type": c["type"], "weight": c["weight"], "active": c["active"],
             "raw_time": c["raw_time"].tolist(), "raw_strain": c["raw_strain"].tolist(),
             "raw_stress": c["raw_stress"].tolist(), "time": c["time"].tolist(),
             "strain": c["strain"].tolist(), "stress": c["stress"].tolist(), "r2": c["r2"]}
            for c in self.case_manager.cases
        ]
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump({"material_parameters": mat_data, "cases": cases_s}, f, ensure_ascii=False, indent=4)
            QMessageBox.information(self, "Saved", "Session saved.")
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))
    
    def load_session(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load Session", "", "JSON (*.json)")
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                sd = json.load(f)
            self.param_editor.set_parameters(sd["material_parameters"])
            self.case_manager.cases.clear()
            for c in sd["cases"]:
                self.case_manager.cases.append({
                    "name": c["name"], "type": c["type"], "weight": c["weight"], "active": c["active"],
                    "raw_time": np.array(c["raw_time"]), "raw_strain": np.array(c["raw_strain"]),
                    "raw_stress": np.array(c["raw_stress"]), "time": np.array(c["time"]),
                    "strain": np.array(c["strain"]), "stress": np.array(c["stress"]), "r2": c["r2"]
                })
            self.case_manager.refresh_table()
            self.on_cases_updated()
            QMessageBox.information(self, "Loaded", "Session restored.")
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))
    
    def generate_html_report_file(self):
        mat_data = self.param_editor.get_parameters()
        if not mat_data:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save Report", "", "HTML (*.html)")
        if not path:
            return
        ReportGenerator.generate_html_report(
            mat_data=mat_data,
            cases=self.case_manager.cases,
            loss_history=[],
            figures_callbacks=[lambda: self.preview_canvas.figure, lambda: self.cal_canvas.figure],
            file_path=path
        )
        QMessageBox.information(self, "Report", "Report generated.")
    
    def run_bayesian_mcmc(self):
        mat_data = self.param_editor.get_parameters()
        active_ds = self.case_manager.get_active_datasets()
        if not active_ds:
            QMessageBox.warning(self, "No Data", "Activate a dataset.")
            return
        ds = active_ds[0]
        if "prony_series" in mat_data:
            fixed_visc = {"stiffness_ratio": mat_data["prony_series"][0]["g_i"], "creep_params": jnp.array([mat_data["prony_series"][0]["tau_i"]])}
        else:
            fixed_visc = {"stiffness_ratio": mat_data["networks"][0]["stiffness_ratio"], "creep_params": mat_data["networks"][0]["creep_params"][:2]}
        self.mcmc_worker = MCMCWorker(times=ds["times"], F_history=ds["F_history"], target_diff=ds["target_diff"], fixed_viscous_params=fixed_visc, num_samples=200, num_warmup=100)
        self.mcmc_worker.finished_success.connect(lambda s: QMessageBox.information(self, "MCMC Done", "Sampling complete."))
        self.mcmc_worker.failed.connect(lambda e: QMessageBox.critical(self, "MCMC Failed", e))
        self.mcmc_worker.start()
    
    def prewarm_jax(self):
        try:
            # Simple JAX warmup — skip complex prewarm to avoid pytree structure mismatches
            jnp.ones(10).sum()  # Minimal JAX call to initialize JIT
        except Exception as e:
            logging.getLogger(__name__).warning("JAX prewarm failed: %s", e)


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setStyleSheet(STYLE_QSS)
    app.setFont(QFont("Segoe UI", 10))
    window = MCalibrationApp()
    window.show()
    QTimer.singleShot(200, window.prewarm_jax)
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
