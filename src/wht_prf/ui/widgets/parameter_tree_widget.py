"""
ParameterTreeWidget — Tree 형식으로 Hyperelastic / Viscoelastic 파라미터를 표시.
각 파라미터: Name (12자) | Min | Value | Max | [Slider Button]
슬라이더는 선택된 항목 아래에 인라인으로 표시됨.
"""
import numpy as np
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTreeWidget, QTreeWidgetItem, QPushButton,
    QLabel, QSlider, QDoubleSpinBox, QHeaderView, QAbstractItemView, QFrame
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont


class ParameterTreeWidget(QWidget):
    """Tree 형식 파라미터 테이블 위젯 (슬라이더 인라인).

    Signals:
        parameter_changed(): 파라미터 값 변경 시
    """
    parameter_changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._params = {}
        self._current_item = None
        self._build_ui()

    def _build_ui(self):
        """UI: Tree + 슬라이더 영역"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        # Tree 위젯
        self._tree = QTreeWidget()
        self._tree.setColumnCount(5)
        self._tree.setHeaderLabels(["Name", "Min", "Value", "Max", "Slider"])
        self._tree.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._tree.setSelectionMode(QAbstractItemView.SingleSelection)
        self._tree.setAlternatingRowColors(True)
        self._tree.setRootIsDecorated(True)

        # 컬럼 너비
        hh = self._tree.header()
        hh.setSectionResizeMode(0, QHeaderView.ResizeToContents)  # Name
        hh.setSectionResizeMode(1, QHeaderView.ResizeToContents)  # Min
        hh.setSectionResizeMode(2, QHeaderView.Stretch)           # Value
        hh.setSectionResizeMode(3, QHeaderView.ResizeToContents)  # Max
        hh.setSectionResizeMode(4, QHeaderView.ResizeToContents)  # Button

        self._tree.clicked.connect(self._on_item_clicked)
        layout.addWidget(self._tree, 1)

        # 슬라이더 영역 (선택되지 않으면 숨김)
        self._slider_frame = QFrame()
        self._slider_frame.setStyleSheet("QFrame { background: #252526; border-top: 1px solid #3c3c3c; }")
        slider_layout = QVBoxLayout(self._slider_frame)
        slider_layout.setContentsMargins(8, 8, 8, 8)
        slider_layout.setSpacing(6)

        # 라벨: 현재 선택 항목
        self._label_selected = QLabel("(선택 없음)")
        self._label_selected.setStyleSheet("color: #cccccc; font-size: 13px;")
        slider_layout.addWidget(self._label_selected)

        # 슬라이더 + 값
        slider_h = QHBoxLayout()
        self._slider = QSlider(Qt.Horizontal)
        self._slider.setRange(0, 1000)
        self._slider.setMinimumWidth(150)
        self._slider.valueChanged.connect(self._on_slider_changed)
        slider_h.addWidget(QLabel("값:"), 0)
        slider_h.addWidget(self._slider, 1)

        self._spinbox = QDoubleSpinBox()
        self._spinbox.setMinimumWidth(100)
        self._spinbox.setMaximumWidth(120)
        self._spinbox.valueChanged.connect(self._on_spinbox_changed)
        slider_h.addWidget(self._spinbox, 0)

        slider_layout.addLayout(slider_h)

        self._slider_frame.setVisible(False)
        self._slider_frame.setMaximumHeight(100)
        layout.addWidget(self._slider_frame, 0)

    def set_parameters(self, mat_data: dict):
        """mat_data를 분석해서 트리 구조로 표시."""
        self._tree.clear()
        self._params = mat_data
        self._current_item = None

        he_type = mat_data.get("hyperelastic_type", "YEOH")
        he_params = mat_data.get("hyperelastic_params", [])
        networks = mat_data.get("networks", [])

        # Hyperelastic 섹션
        he_item = QTreeWidgetItem(["Hyperelastic", "", "", "", ""])
        he_item.setFont(0, QFont("Segoe UI", 10, QFont.Bold))
        he_item.setExpanded(True)
        self._tree.addTopLevelItem(he_item)

        # HE 파라미터들
        he_params_spec = [
            ("C10", 0.001, 1000.0),
            ("C20", -10.0, 10.0),
            ("C30", -10.0, 10.0),
        ]
        for i, (name, min_v, max_v) in enumerate(he_params_spec):
            val = he_params[i] if i < len(he_params) else 0.0
            self._add_param_row(he_item, name, val, min_v, max_v)

        # Viscoelastic 섹션
        ve_item = QTreeWidgetItem(["Viscoelastic", "", "", "", ""])
        ve_item.setFont(0, QFont("Segoe UI", 10, QFont.Bold))
        ve_item.setExpanded(True)
        self._tree.addTopLevelItem(ve_item)

        # Network별 파라미터
        for net_idx, net in enumerate(networks):
            net_item = QTreeWidgetItem([f"Network {net_idx + 1}", "", "", "", ""])
            net_item.setFont(0, QFont("Segoe UI", 9, QFont.Bold))
            net_item.setExpanded(True)
            ve_item.addChild(net_item)

            # SRATIO
            s_val = net.get("stiffness_ratio", 0.1)
            self._add_param_row(net_item, "SRATIO", s_val, 0.001, 0.99)

            # Creep params
            c_params = net.get("creep_params", [])
            creep_names = [("A", 1e-30, 1e2), ("n", 1.0, 20.0), ("m", -1.5, 0.0)]
            for i, (c_name, c_min, c_max) in enumerate(creep_names):
                c_val = c_params[i] if i < len(c_params) else 0.0
                self._add_param_row(net_item, c_name, c_val, c_min, c_max)

    def _add_param_row(self, parent_item: QTreeWidgetItem, name: str, value: float, min_v: float, max_v: float):
        """파라미터 행 추가"""
        is_log = "A" in name or "tau" in name

        # 값 형식
        val_str = f"{value:.5E}" if is_log else f"{value:.4f}"
        min_str = f"{min_v:.5E}" if is_log else f"{min_v:.4f}"
        max_str = f"{max_v:.5E}" if is_log else f"{max_v:.4f}"

        item = QTreeWidgetItem([name, min_str, val_str, max_str, "📊"])
        item.setData(0, Qt.UserRole, {
            "name": name,
            "value": value,
            "min": min_v,
            "max": max_v,
            "is_log": is_log
        })
        parent_item.addChild(item)

    def _on_item_clicked(self, index):
        """아이템 클릭: 슬라이더 영역 표시"""
        if index.column() != 4:  # 슬라이더 버튼이 아니면
            return

        item = self._tree.itemFromIndex(index)
        if item is None or item.parent() is None:
            return  # 부모 노드는 무시

        data = item.data(0, Qt.UserRole)
        if not data:
            return

        # 슬라이더 설정
        self._current_item = item
        self._update_slider(data)
        self._slider_frame.setVisible(True)

    def _update_slider(self, data: dict):
        """슬라이더를 파라미터 데이터에 맞게 업데이트"""
        self._slider.blockSignals(True)
        self._spinbox.blockSignals(True)

        name = data["name"]
        value = data["value"]
        min_v = data["min"]
        max_v = data["max"]
        is_log = data["is_log"]

        self._label_selected.setText(f"선택: {name}")

        self._spinbox.setRange(min_v, max_v)
        self._spinbox.setValue(value)

        # 슬라이더 위치 계산
        if is_log:
            log_min = np.log10(max(min_v, 1e-35))
            log_max = np.log10(max(max_v, 1e-5))
            log_val = np.log10(max(value, 1e-35))
            pos = int((log_val - log_min) / (log_max - log_min + 1e-12) * 1000)
        else:
            pos = int((value - min_v) / (max_v - min_v + 1e-12) * 1000)

        self._slider.setValue(max(0, min(1000, pos)))

        self._slider.blockSignals(False)
        self._spinbox.blockSignals(False)

    def _on_slider_changed(self, pos: int):
        """슬라이더 변경"""
        if self._current_item is None:
            return

        data = self._current_item.data(0, Qt.UserRole)
        min_v = data["min"]
        max_v = data["max"]
        is_log = data["is_log"]

        # 값 계산
        if is_log:
            log_min = np.log10(max(min_v, 1e-35))
            log_max = np.log10(max(max_v, 1e-5))
            value = 10 ** (log_min + (log_max - log_min) * (pos / 1000.0))
        else:
            value = min_v + (max_v - min_v) * (pos / 1000.0)

        self._spinbox.blockSignals(True)
        self._spinbox.setValue(value)
        self._spinbox.blockSignals(False)

    def _on_spinbox_changed(self, value: float):
        """스핀박스 변경"""
        if self._current_item is None:
            return

        data = self._current_item.data(0, Qt.UserRole)

        # Tree의 Value 컬럼 업데이트
        is_log = data["is_log"]
        val_str = f"{value:.5E}" if is_log else f"{value:.4f}"
        self._current_item.setText(2, val_str)

        # 데이터 업데이트
        data["value"] = value
        self._current_item.setData(0, Qt.UserRole, data)

        # 신호 발생
        self.parameter_changed.emit()

    def get_parameters(self) -> dict:
        """현재 파라미터 반환"""
        return self._params
