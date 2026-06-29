"""
SegmentBuilderWidget — MCalibration의 Virtual Experiment (Segments) UI.
Ramp / Hold / Unload / Cycle 세그먼트를 CRUD 테이블로 구성하고
하중 이력을 실시간 프리뷰한다.
"""
import numpy as np
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox,
    QPushButton, QTableWidget, QTableWidgetItem, QHeaderView,
    QDialog, QDialogButtonBox, QFormLayout, QDoubleSpinBox,
    QSpinBox, QComboBox, QLabel, QMenu, QAbstractItemView,
)
from PySide6.QtCore import Signal, Qt
from PySide6.QtGui import QColor

from wht_prf.core.load_case import Segment, SEGMENT_MODES
from wht_prf.ui.widgets.plot_canvas import PlotCanvas


# ── 세그먼트 편집 다이얼로그 ─────────────────────────────────────────────────

class SegmentEditDialog(QDialog):
    """단일 Segment를 편집하는 간단한 폼 다이얼로그."""

    def __init__(self, seg: Segment | None = None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Edit segment")
        self.setMinimumWidth(300)
        self._seg = seg or Segment()
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)

        form = QFormLayout()
        form.setSpacing(8)

        self.cmb_mode = QComboBox()
        self.cmb_mode.addItems(SEGMENT_MODES)
        self.cmb_mode.setCurrentText(self._seg.mode)
        self.cmb_mode.currentTextChanged.connect(self._on_mode_changed)
        form.addRow("Mode", self.cmb_mode)

        self.spn_rate = QDoubleSpinBox()
        self.spn_rate.setRange(1e-6, 1000.0)
        self.spn_rate.setDecimals(4)
        self.spn_rate.setSingleStep(0.001)
        self.spn_rate.setValue(self._seg.rate)
        self.spn_rate.setSuffix(" /s")
        form.addRow("Strain rate", self.spn_rate)

        self.spn_strain = QDoubleSpinBox()
        self.spn_strain.setRange(-10.0, 10.0)
        self.spn_strain.setDecimals(4)
        self.spn_strain.setSingleStep(0.01)
        self.spn_strain.setValue(self._seg.end_strain)
        form.addRow("End strain", self.spn_strain)

        self.spn_dur = QDoubleSpinBox()
        self.spn_dur.setRange(0.0, 1e7)
        self.spn_dur.setDecimals(1)
        self.spn_dur.setSingleStep(10.0)
        self.spn_dur.setValue(self._seg.duration)
        self.spn_dur.setSuffix(" s")
        form.addRow("Hold duration", self.spn_dur)

        self.spn_cycles = QSpinBox()
        self.spn_cycles.setRange(1, 1000)
        self.spn_cycles.setValue(self._seg.cycles)
        form.addRow("Cycles", self.spn_cycles)

        self.spn_temp = QDoubleSpinBox()
        self.spn_temp.setRange(77.0, 1500.0)
        self.spn_temp.setDecimals(1)
        self.spn_temp.setValue(self._seg.temperature)
        self.spn_temp.setSuffix(" K")
        form.addRow("Temperature", self.spn_temp)

        layout.addLayout(form)

        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)

        self._on_mode_changed(self._seg.mode)

    def _on_mode_changed(self, mode: str):
        is_hold  = mode == "Hold"
        is_cycle = mode == "Cycle"
        is_ramp  = mode in ("Ramp", "Unload", "Rate Jump")
        self.spn_dur.setEnabled(is_hold)
        self.spn_cycles.setEnabled(is_cycle)
        self.spn_rate.setEnabled(not is_hold)
        self.spn_strain.setEnabled(not (mode == "Rate Jump" or is_hold))

    def get_segment(self) -> Segment:
        return Segment(
            mode        = self.cmb_mode.currentText(),
            end_strain  = self.spn_strain.value(),
            rate        = self.spn_rate.value(),
            duration    = self.spn_dur.value(),
            cycles      = self.spn_cycles.value(),
            temperature = self.spn_temp.value(),
        )


# ── 메인 위젯 ────────────────────────────────────────────────────────────────

class SegmentBuilderWidget(QWidget):
    """Ramp / Hold / Unload / Cycle 세그먼트를 편집하고
    가상 하중 이력을 실시간 프리뷰하는 위젯.

    Signals:
        segments_changed(list): 세그먼트 목록이 변경될 때 발생
    """
    segments_changed = Signal(list)

    _HEADERS = ["#", "Mode", "Definition", "End time (s)"]
    _COL_COLORS = {
        "Ramp":      "#2a4a7f",
        "Hold":      "#2a5a3a",
        "Unload":    "#5a3a2a",
        "Cycle":     "#4a2a6a",
        "Rate Jump": "#3a3a3a",
    }

    def __init__(self, parent=None):
        super().__init__(parent)
        self._segments: list[Segment] = []
        self._build_ui()

    # ── UI 구성 ──────────────────────────────────────────────────────────────

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(6)

        # 툴바
        bar = QHBoxLayout()
        self.btn_add = QPushButton("+ Add ▾")
        self.btn_add.setFixedWidth(90)
        self.btn_add.clicked.connect(self._show_add_menu)

        self.btn_edit   = QPushButton("Edit")
        self.btn_remove = QPushButton("Remove")
        self.btn_up     = QPushButton("↑")
        self.btn_down   = QPushButton("↓")
        for b in (self.btn_edit, self.btn_remove, self.btn_up, self.btn_down):
            b.setFixedWidth(64)

        self.btn_edit.clicked.connect(self._on_edit)
        self.btn_remove.clicked.connect(self._on_remove)
        self.btn_up.clicked.connect(lambda: self._on_move(-1))
        self.btn_down.clicked.connect(lambda: self._on_move(1))

        bar.addWidget(self.btn_add)
        bar.addWidget(self.btn_edit)
        bar.addWidget(self.btn_remove)
        bar.addWidget(self.btn_up)
        bar.addWidget(self.btn_down)
        bar.addStretch()
        root.addLayout(bar)

        # 테이블
        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(self._HEADERS)
        # 열폭 조절 가능하게 설정
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        self.table.horizontalHeader().setStretchLastSection(False)
        # 스크롤 지원
        self.table.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.table.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setAlternatingRowColors(False)
        self.table.verticalHeader().setVisible(False)
        self.table.doubleClicked.connect(self._on_edit)
        self.table.setFixedHeight(160)
        root.addWidget(self.table)

        # 프리뷰 플롯
        grp = QGroupBox("Load history preview")
        grp.setStyleSheet("QGroupBox { font-size: 11px; }")
        glayout = QVBoxLayout(grp)
        glayout.setContentsMargins(4, 4, 4, 4)
        self._plot = PlotCanvas(self, width=4, height=2.0, dpi=90)
        glayout.addWidget(self._plot)
        root.addWidget(grp)

    # ── 슬롯 ─────────────────────────────────────────────────────────────────

    def _show_add_menu(self):
        menu = QMenu(self)
        for mode in SEGMENT_MODES:
            menu.addAction(mode, lambda m=mode: self._on_add(m))
        menu.exec(self.btn_add.mapToGlobal(self.btn_add.rect().bottomLeft()))

    def _on_add(self, mode: str):
        seg = Segment(mode=mode)
        dlg = SegmentEditDialog(seg, self)
        if dlg.exec() == QDialog.Accepted:
            self._segments.append(dlg.get_segment())
            self._refresh_table()
            self._refresh_preview()
            self.segments_changed.emit(self._segments)

    def _on_edit(self):
        row = self.table.currentRow()
        if row < 0 or row >= len(self._segments):
            return
        dlg = SegmentEditDialog(self._segments[row], self)
        if dlg.exec() == QDialog.Accepted:
            self._segments[row] = dlg.get_segment()
            self._refresh_table()
            self._refresh_preview()
            self.segments_changed.emit(self._segments)

    def _on_remove(self):
        row = self.table.currentRow()
        if 0 <= row < len(self._segments):
            self._segments.pop(row)
            self._refresh_table()
            self._refresh_preview()
            self.segments_changed.emit(self._segments)

    def _on_move(self, delta: int):
        row = self.table.currentRow()
        new = row + delta
        if 0 <= new < len(self._segments):
            self._segments[row], self._segments[new] = self._segments[new], self._segments[row]
            self._refresh_table()
            self.table.selectRow(new)
            self._refresh_preview()
            self.segments_changed.emit(self._segments)

    # ── 내부 갱신 ────────────────────────────────────────────────────────────

    def _refresh_table(self):
        self.table.setRowCount(0)
        t_cur = 0.0
        prev_eps = 0.0
        for i, seg in enumerate(self._segments):
            dur = seg.duration_s(prev_eps)
            t_cur += dur
            if seg.mode in ("Ramp", "Unload"):
                prev_eps = seg.end_strain

            items = [
                str(i + 1),
                seg.mode,
                seg.label(),
                f"{t_cur:.1f}",
            ]
            row = self.table.rowCount()
            self.table.insertRow(row)
            color = QColor(self._COL_COLORS.get(seg.mode, "#333"))
            for col, text in enumerate(items):
                it = QTableWidgetItem(text)
                it.setForeground(QColor("#dcdcdc"))
                it.setBackground(color)
                self.table.setItem(row, col, it)

    def _refresh_preview(self):
        from wht_prf.ui.simulation import segments_to_time_strain
        if not self._segments:
            ax = self._plot.axes
            ax.cla()
            ax.set_xlabel("Time (s)", fontsize=8)
            ax.set_ylabel("Strain", fontsize=8)
            ax.tick_params(labelsize=7)
            self._plot.draw()
            return
        try:
            times, strains = segments_to_time_strain(self._segments, dt_max=0.5)
            ax = self._plot.axes
            ax.cla()
            ax.plot(times, strains, color="#4a9eff", linewidth=1.4)
            ax.set_xlabel("Time (s)", fontsize=8)
            ax.set_ylabel("Strain", fontsize=8)
            ax.tick_params(labelsize=7)
            ax.grid(True, linewidth=0.4, alpha=0.4)
            self._plot.draw()
        except Exception:
            pass

    # ── 공개 API ─────────────────────────────────────────────────────────────

    def get_segments(self) -> list:
        return list(self._segments)

    def set_segments(self, segments: list):
        self._segments = list(segments)
        self._refresh_table()
        self._refresh_preview()
