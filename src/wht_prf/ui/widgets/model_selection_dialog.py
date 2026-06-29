"""
ModelSelectionDialog — 초탄성 모델 & 점탄성 모델 선택 + 지능형 추천.
아이콘 버튼으로 팝업되는 다이얼로그로, 모델 설명 HTML 뷰와
데이터셋 기반 자동 추천 기능을 포함합니다.
"""
import numpy as np
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QGroupBox, QFormLayout,
    QRadioButton, QButtonGroup, QLabel, QSpinBox, QPushButton,
    QDialogButtonBox, QTextBrowser, QMessageBox, QComboBox,
)
from PySide6.QtCore import Qt, Signal


# ── 모델 설명 (HTML/Markdown) ────────────────────────────────────────────

MODEL_DESCRIPTIONS = {
    "YEOH": """
<h3>Yeoh Hyperelastic Model</h3>
<p><strong>용도:</strong> 고무, 엘라스토머 재료</p>
<p><strong>특징:</strong></p>
<ul>
<li>3개 파라미터 (C10, C20, C30)로 간단하고 빠른 계산</li>
<li>무압축성(incompressible) 가정</li>
<li>중간 변형률 범위에서 정확 (보통 0-200%)</li>
<li>대변형에서는 부정확할 수 있음</li>
</ul>
<p><strong>파라미터:</strong> C10, C20, C30, D1 (압축성)</p>
""",
    "NEO_HOOKEAN": """
<h3>Neo-Hookean Hyperelastic Model</h3>
<p><strong>용도:</strong> 간단한 엘라스토머, 시작 모델</p>
<p><strong>특징:</strong></p>
<ul>
<li>가장 단순한 모델 (1개 파라미터 C10)</li>
<li>계산이 매우 빠름</li>
<li>저~중간 변형률에서만 적용</li>
<li>실제 데이터와 맞추기 어려움</li>
</ul>
<p><strong>파라미터:</strong> C10, D1 (압축성)</p>
""",
    "ARRUDA_BOYCE": """
<h3>Arruda-Boyce (Limited Stretch) Model</h3>
<p><strong>용도:</strong> 고변형 엘라스토머, 사슬 한계 고려</p>
<p><strong>특징:</strong></p>
<ul>
<li>폴리머 사슬의 신축 한계를 고려 (물리적 토대)</li>
<li>대변형 (200% 이상)에서 우수</li>
<li>2개 파라미터 (μ, λ_L)</li>
<li>비선형성 강한 재료에 추천</li>
</ul>
<p><strong>파라미터:</strong> μ, λ_L, D1 (압축성)</p>
""",
    "OGDEN": """
<h3>Ogden Hyperelastic Model</h3>
<p><strong>용도:</strong> 복잡한 비선형 거동, 고정확도 필요</p>
<p><strong>특징:</strong></p>
<ul>
<li>최대 6개 항 (각 항마다 μ_i, α_i) 지원</li>
<li>넓은 변형률 범위에서 높은 정확도</li>
<li>계산 비용 증가</li>
<li>고급 사용자 대상</li>
</ul>
<p><strong>파라미터:</strong> μ₁, α₁, μ₂, α₂, ... (최대 6개 항)</p>
""",
    "VAN_DER_WAALS": """
<h3>Van der Waals Hyperelastic Model</h3>
<p><strong>용도:</strong> 입체 장애 효과 고려, 온도 민감 재료</p>
<p><strong>특징:</strong></p>
<ul>
<li>분자간 상호작용 (입체 장애) 모델링</li>
<li>온도 의존성 포함 가능</li>
<li>3개 파라미터 (μ, λ_m, a, β)</li>
<li>특수 재료에만 추천</li>
</ul>
<p><strong>파라미터:</strong> μ, λ_m, a, β, D</p>
""",
}

MODEL_MODE_DESCRIPTIONS = {
    "PRF": """
<h4>PRF (Parallel Rheological Framework)</h4>
<p><strong>구조:</strong> 초탄성 + 여러 N개 네트워크 (크리프/이완)</p>
<p><strong>특징:</strong></p>
<ul>
<li>각 네트워크는 독립적 크리프 법칙 (Time/Strain/Bergstrom 등)</li>
<li>복잡한 점탄성 거동 표현</li>
<li>네트워크 수는 데이터 복잡도에 따라 1~3개 추천</li>
</ul>
""",
    "PRONY": """
<h4>Prony Series (고전 점탄성)</h4>
<p><strong>구조:</strong> 초탄성 + 여러 Prony 항</p>
<p><strong>특징:</strong></p>
<ul>
<li>선형 점탄성 이론 기반</li>
<li>작은 변형에서 정확</li>
<li>계산이 빠름</li>
<li>대변형 재료에는 부적합</li>
</ul>
""",
}


class ModelSelectionDialog(QDialog):
    """모델 타입, 모드 선택 + 지능형 추천 다이얼로그.

    Signals:
        model_selected(he_type, model_mode, num_networks): 모델 선택 완료
    """
    model_selected = Signal(str, str, int)  # (he_type, model_mode, num_networks)

    def __init__(self, parent=None, active_datasets: list | None = None):
        super().__init__(parent)
        self.setWindowTitle("Model Selection & Recommendation")
        self.setMinimumWidth(900)
        self.setMinimumHeight(600)
        self._active_datasets = active_datasets or []
        self._selected_he_type = "YEOH"
        self._selected_model_mode = "PRF"
        self._selected_num_networks = 1
        # Initialize widgets to None before _build_ui
        self.text_browser = None
        self.spn_networks = None
        self.btn_recommend = None
        self._build_ui()

    # ── UI 구성 ──────────────────────────────────────────────────────────────

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(8)

        # ── 2열 레이아웃: 좌측(선택), 우측(설명) ──
        main_h = QHBoxLayout()
        main_h.setSpacing(12)

        # 좌측: 모델 선택 패널
        left_panel = self._build_selection_panel()
        main_h.addWidget(left_panel, 1)

        # 우측: 모델 설명 HTML 뷰어
        self.text_browser = QTextBrowser()
        self.text_browser.setHtml(MODEL_DESCRIPTIONS.get("YEOH", ""))
        main_h.addWidget(self.text_browser, 1)

        root.addLayout(main_h, 1)

        # ── 하단: 버튼 ──
        btn_layout = QHBoxLayout()
        self.btn_recommend = QPushButton("🤖 Recommend from Dataset")
        self.btn_recommend.clicked.connect(self._on_recommend)
        self.btn_recommend.setEnabled(len(self._active_datasets) > 0)
        btn_layout.addWidget(self.btn_recommend)
        btn_layout.addStretch()

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        btn_layout.addWidget(buttons)

        root.addLayout(btn_layout)

    def _build_selection_panel(self) -> QGroupBox:
        grp = QGroupBox("Model Selection")
        form = QFormLayout(grp)
        form.setSpacing(8)

        # ── Hyperelastic Type ──
        he_group = QButtonGroup()
        he_layout = QHBoxLayout()
        for i, he_type in enumerate(["YEOH", "NEO_HOOKEAN", "ARRUDA_BOYCE", "OGDEN", "VAN_DER_WAALS"]):
            rb = QRadioButton(he_type.replace("_", " "))
            rb.toggled.connect(lambda checked, t=he_type: self._on_he_type_changed(t) if (checked and self.text_browser) else None)
            he_group.addButton(rb, i)
            he_layout.addWidget(rb)
            if he_type == "YEOH":
                rb.setChecked(True)

        form.addRow("Hyperelastic Type:", he_layout)

        # ── Model Mode ──
        mode_group = QButtonGroup()
        mode_layout = QHBoxLayout()
        for i, mode in enumerate(["PRF", "PRONY"]):
            rb = QRadioButton(mode)
            rb.toggled.connect(lambda checked, m=mode: self._on_mode_changed(m) if (checked and self.spn_networks) else None)
            mode_group.addButton(rb, i)
            mode_layout.addWidget(rb)
            if mode == "PRF":
                rb.setChecked(True)

        form.addRow("Model Mode:", mode_layout)

        # ── Number of Networks (PRF only) ──
        self.spn_networks = QSpinBox()
        self.spn_networks.setRange(1, 5)
        self.spn_networks.setValue(1)
        self.spn_networks.valueChanged.connect(self._on_networks_changed)
        form.addRow("Networks (PRF):", self.spn_networks)

        return grp

    # ── 슬롯 ─────────────────────────────────────────────────────────────────

    def _on_he_type_changed(self, he_type: str):
        """초탄성 타입 변경 → 설명 업데이트"""
        self._selected_he_type = he_type
        html = MODEL_DESCRIPTIONS.get(he_type, "")
        html += "<hr/>" + MODEL_MODE_DESCRIPTIONS.get(self._selected_model_mode, "")
        self.text_browser.setHtml(html)

    def _on_mode_changed(self, mode: str):
        """모델 모드 변경 → 설명 업데이트"""
        self._selected_model_mode = mode
        self.spn_networks.setEnabled(mode == "PRF")
        html = MODEL_DESCRIPTIONS.get(self._selected_he_type, "")
        html += "<hr/>" + MODEL_MODE_DESCRIPTIONS.get(mode, "")
        self.text_browser.setHtml(html)

    def _on_networks_changed(self, value: int):
        """네트워크 수 변경"""
        self._selected_num_networks = value

    def _on_recommend(self):
        """데이터셋 기반 지능형 모델 추천"""
        if not self._active_datasets:
            QMessageBox.information(self, "No Data", "Load cases를 먼저 추가하세요.")
            return

        he_type, mode, num_nets = self._recommend_model()
        self.setWindowTitle(f"✓ Recommended: {he_type} + {mode}")
        QMessageBox.information(
            self,
            "Recommendation",
            f"추천 모델:\n- Hyperelastic: {he_type}\n- Mode: {mode}\n- Networks: {num_nets}",
        )

    def _recommend_model(self) -> tuple[str, str, int]:
        """데이터셋 분석하여 모델 추천.

        Returns:
            (he_type, model_mode, num_networks)
        """
        # 데이터셋 특성 분석
        stress_ranges = []
        relaxation_times = []
        is_creep_data = False

        for ds in self._active_datasets:
            # 응력 범위
            if "F_history" in ds and len(ds["F_history"]) > 0:
                stress_approx = np.max(np.abs(ds["F_history"])) - 1.0  # 변형도
                stress_ranges.append(abs(stress_approx))

            # 이완/크리프 시간 추정
            if "times" in ds and len(ds["times"]) > 1:
                max_time = ds["times"][-1]
                relaxation_times.append(max_time)

            # 크리프 데이터 판정
            lc_type = ds.get("lc_type", "")
            if lc_type in ("stress_relax", "creep"):
                is_creep_data = True

        # 휴리스틱 기반 추천
        avg_stress = np.mean(stress_ranges) if stress_ranges else 0.1
        max_time = max(relaxation_times) if relaxation_times else 10.0

        # 응력 범위에 따른 모델 선택
        if avg_stress > 2.0:
            he_type = "ARRUDA_BOYCE"  # 대변형
        elif avg_stress > 1.0:
            he_type = "YEOH"  # 중간 변형
        else:
            he_type = "NEO_HOOKEAN"  # 소변형

        # 크리프 데이터 → PRF, 아니면 PRONY
        mode = "PRF" if is_creep_data else "PRONY"

        # 네트워크/항 수: 최대 시간 기반
        if max_time > 100:
            num_nets = 3
        elif max_time > 10:
            num_nets = 2
        else:
            num_nets = 1

        return he_type, mode, num_nets

    # ── 공개 API ─────────────────────────────────────────────────────────────

    def set_datasets(self, datasets: list):
        """활성 데이터셋 설정 → 추천 버튼 활성화"""
        self._active_datasets = datasets
        self.btn_recommend.setEnabled(len(datasets) > 0)

    def get_model_config(self) -> tuple[str, str, int]:
        """선택된 모델 구성 반환: (he_type, model_mode, num_networks)"""
        return self._selected_he_type, self._selected_model_mode, self._selected_num_networks
