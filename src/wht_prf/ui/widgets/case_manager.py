# WHTOOLs MATCALIB 2026
"""
Experimental Case Manager Widget.
Allows importing multiple CSV/Excel test datasets, preprocessing them 
(offset correction, log/linear downsampling, Cauchy stress conversion), 
and selecting which cases to include in calibration.
"""
import os
import pandas as pd
import numpy as np
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QTableWidget, QTableWidgetItem, 
    QFileDialog, QHeaderView, QCheckBox, QDoubleSpinBox, QComboBox, QLabel, QSpinBox, QMessageBox
)
from PySide6.QtCore import Signal, Qt
from wht_prf.core.managers import SessionManager

class CaseManagerWidget(QWidget):
    """실험 데이터 파일들을 로드하고 가중치, 전처리(오프셋, 다운샘플링, 참변형률 환산) 
    및 최적화 포함 여부를 직관적으로 매핑·제어하는 위젯 클래스입니다.
    """
    cases_updated = Signal()  # 케이스 목록 또는 데이터 변경 시 발생하는 시그널

    def __init__(self, parent=None):
        super().__init__(parent)
        self.cases = []  # 로드된 데이터 케이스들의 딕셔너리 리스트
        
        # UI 레이아웃 구축
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # 1. 상단 액션 버튼 영역 (간소화: + 버튼만)
        btn_layout = QHBoxLayout()
        self.btn_load = QPushButton("+ 파일 추가")
        self.btn_load.setFixedWidth(100)
        self.btn_load.clicked.connect(self.on_load_files)
        btn_layout.addWidget(self.btn_load)
        btn_layout.addStretch()  # 나머지 공간

        layout.addLayout(btn_layout)
        
        # 2. 케이스 목록 테이블
        self.table = QTableWidget()
        self.table.setColumnCount(7)
        self.table.setHorizontalHeaderLabels([
            "피팅 포함", "케이스 명칭", "시험 종류", "최대 변형률/시간", "가중치 (Weight)", "R² 결과", "삭제"
        ])
        # 열폭 조절 가능하게 설정
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        self.table.horizontalHeader().setStretchLastSection(False)
        # 스크롤 지원
        self.table.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.table.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.table.verticalHeader().setVisible(False)

        layout.addWidget(self.table)
        
    def on_load_files(self):
        """다중 데이터 파일을 열어 로드합니다."""
        files, _ = QFileDialog.getOpenFileNames(
            self, "실험 데이터 로드", "", 
            "Data Files (*.csv *.txt *.xlsx);;All Files (*)"
        )
        
        if not files:
            return
            
        for path in files:
            self.load_single_file(path)
            
        self.refresh_table()
        self._sync_with_backend()
        self.cases_updated.emit()

    def load_single_file(self, path):
        """단일 텍스트 또는 엑셀 파일을 읽어 케이스 리스트에 추가합니다."""
        filename = os.path.basename(path)
        try:
            # 엑셀 파일인 경우 다중 시트 파싱 지원
            if path.endswith('.xlsx'):
                xl = pd.ExcelFile(path)
                for sheet_name in xl.sheet_names:
                    df = xl.parse(sheet_name)
                    self.parse_and_append_df(df, f"{filename} ({sheet_name})")
            else:
                # 콤마, 공백 구분자 자동 판단
                with open(path, 'r', encoding='utf-8') as f:
                    first_line = f.readline()
                sep = ',' if ',' in first_line else r'\s+'
                df = pd.read_csv(path, sep=sep, header=None, comment='#')
                self.parse_and_append_df(df, filename)
        except Exception as e:
            QMessageBox.critical(self, "파일 로드 에러", f"{filename}을 불러오지 못했습니다.\n에러: {str(e)}")

    def parse_and_append_df(self, df, case_name):
        # 헤더나 쓰레기값 무시하고 첫 3개 열 추출
        df_clean = df.dropna().values
        if len(df_clean) < 3:
            raise ValueError("데이터의 유효 데이터 행 수가 너무 적습니다.")
            
        # 첫 3열 바인딩: 시간, 응력, 변형률 순서 파악 (텍스트 포맷: time, stress, strain 또는 time, strain, stress)
        # test_prony_val_extracted.txt 은 time, stress, strain 임.
        # step11 아바쿠스 검증 데이터는 time, strain, stress 순서일 수 있음.
        # 데이터 경향성 자동 감지: strain은 변위 성분이므로 절댓값이 대개 0 ~ 2 이내임.
        # stress는 대개 수십~수천 MPa/Pa 스케일임.
        col0 = pd.to_numeric(df_clean[:, 0], errors='coerce')
        col1 = pd.to_numeric(df_clean[:, 1], errors='coerce')
        col2 = pd.to_numeric(df_clean[:, 2], errors='coerce')
        
        time = np.array(col0)
        
        # col1과 col2 중 어느 것이 strain인지 판별
        max_c1 = np.max(np.abs(col1))
        max_c2 = np.max(np.abs(col2))
        
        if max_c1 < max_c2:
            # col1이 변형률, col2가 응력
            strain = np.array(col1)
            stress = np.array(col2)
        else:
            # col2가 변형률, col1이 응력
            strain = np.array(col2)
            stress = np.array(col1)
            
        # 스케일 오류 자동 정규화 경고 (예: Stress가 Pa 스케일인 경우 1e-6 곱해 MPa로 변환)
        max_stress = np.max(np.abs(stress))
        if max_stress > 1e4:
            reply = QMessageBox.question(
                self, "단위 환산 경고", 
                f"[{case_name}] 응력의 최대값({max_stress:.1f})이 대단히 큽니다. Pa 단위를 MPa 단위로 자동 환산할까요?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes
            )
            if reply == QMessageBox.Yes:
                stress = stress * 1e-6
                
        # 기본 메타데이터 및 raw 복사본 저장
        # 완화(Relaxation) 시험 판별: 마지막 변형률이 0이 아니고 중간과 끝 변형률 변화가 거의 없는 경우
        test_type = "Relaxation" if np.max(time) > 10.0 and np.std(strain[-10:]) < 0.005 else "Uniaxial"
        
        case_data = {
            "name": case_name.split('.')[0],
            "type": test_type,
            "weight": 1.0,
            "active": True,
            "raw_time": time,
            "raw_strain": strain,
            "raw_stress": stress,
            "time": time.copy(),
            "strain": strain.copy(),
            "stress": stress.copy(),
            "r2": None
        }
        self.cases.append(case_data)

    def refresh_table(self):
        """테이블 위젯을 현재 cases 리스트 정보로 다시 렌더링합니다."""
        self.table.setRowCount(len(self.cases))
        for idx, case in enumerate(self.cases):
            # 0. 피팅 포함 체크박스
            chk = QCheckBox()
            chk.setChecked(case["active"])
            chk.stateChanged.connect(lambda _, i=idx, c=chk: self.on_active_changed(i, c.isChecked()))
            self.table.setCellWidget(idx, 0, chk)
            
            # 1. 케이스 명칭
            self.table.setItem(idx, 1, QTableWidgetItem(case["name"]))
            
            # 2. 시험 종류
            cmb = QComboBox()
            cmb.addItems(["Uniaxial", "Relaxation"])
            cmb.setCurrentText(case["type"])
            cmb.currentTextChanged.connect(lambda val, i=idx: self.on_type_changed(i, val))
            self.table.setCellWidget(idx, 2, cmb)
            
            # 3. 최대 변형률 및 시간 정보 표시
            max_strain = np.max(np.abs(case["strain"]))
            max_time = np.max(case["time"])
            self.table.setItem(idx, 3, QTableWidgetItem(f"Strain: {max_strain:.2f} / Time: {max_time:.1f}s"))
            
            # 4. 가중치 설정 스핀박스
            spin = QDoubleSpinBox()
            spin.setRange(0.0, 100.0)
            spin.setValue(case["weight"])
            spin.setSingleStep(0.1)
            spin.valueChanged.connect(lambda val, i=idx: self.on_weight_changed(i, val))
            self.table.setCellWidget(idx, 4, spin)
            
            # 5. R2 통계 결과 표시
            r2_val = case["r2"]
            r2_str = f"{r2_val:.5f}" if r2_val is not None else "-"
            self.table.setItem(idx, 5, QTableWidgetItem(r2_str))
            
            # 6. 삭제 버튼
            btn_del = QPushButton("Delete")
            btn_del.clicked.connect(lambda _, i=idx: self.delete_case(i))
            self.table.setCellWidget(idx, 6, btn_del)

    def on_active_changed(self, idx, val):
        self.cases[idx]["active"] = val
        self._sync_with_backend()
        self.cases_updated.emit()

    def on_type_changed(self, idx, val):
        self.cases[idx]["type"] = val
        self._sync_with_backend()
        self.cases_updated.emit()

    def on_weight_changed(self, idx, val):
        self.cases[idx]["weight"] = val
        self._sync_with_backend()

    def delete_case(self, idx):
        self.cases.pop(idx)
        self.refresh_table()
        self._sync_with_backend()
        self.cases_updated.emit()

    def apply_zero_offset(self):
        """선택 활성화된 케이스들에 대해 원점 정렬(Zero-offset)을 적용합니다."""
        for case in self.cases:
            if case["active"]:
                # 원점 보정 진행 (t0, e0, s0 감산)
                case["time"] = case["time"] - case["time"][0]
                case["strain"] = case["strain"] - case["strain"][0]
                case["stress"] = case["stress"] - case["stress"][0]
        self.refresh_table()
        self._sync_with_backend()
        self.cases_updated.emit()

    def apply_downsampling(self):
        """활성화된 데이터셋의 포인트 밀도를 균등 다운샘플링합니다."""
        target_n = self.spin_downsample.value()
        for case in self.cases:
            if case["active"]:
                n_total = len(case["raw_time"])
                if n_total <= target_n:
                    continue
                # 선형 인덱스 추출을 통해 등간격 다운샘플링 수행
                indices = np.linspace(0, n_total - 1, target_n, dtype=int)
                case["time"] = case["raw_time"][indices]
                case["strain"] = case["raw_strain"][indices]
                case["stress"] = case["raw_stress"][indices]
                
                # 만약 offset 보정이 되어있었던 상태면 다운샘플링 후에도 유지되도록 처리
                case["time"] = case["time"] - case["time"][0]
                case["strain"] = case["strain"] - case["strain"][0]
                case["stress"] = case["stress"] - case["stress"][0]
                
        self.refresh_table()
        self._sync_with_backend()
        self.cases_updated.emit()

    def on_large_strain_toggled(self, state):
        """Eng strain/stress <-> True strain/stress 변환을 적용합니다."""
        is_true = self.chk_large_strain.isChecked()
        for case in self.cases:
            # raw copy로부터 변환 수행
            eng_strain = case["raw_strain"]
            eng_stress = case["raw_stress"]
            
            if is_true:
                # True Strain = ln(1 + e)
                # True Stress = s * (1 + e)
                case["strain"] = np.log(1.0 + eng_strain)
                case["stress"] = eng_stress * (1.0 + eng_strain)
            else:
                case["strain"] = eng_strain.copy()
                case["stress"] = eng_stress.copy()
                
        self.refresh_table()
        self._sync_with_backend()
        self.cases_updated.emit()

    def _sync_with_backend(self):
        """내부 상태를 SessionManager 백엔드로 덮어씌워 동기화합니다."""
        active_list = self.get_active_datasets()
        SessionManager.get_instance().data_manager.set_test_cases(active_list)

    def get_active_datasets(self) -> list:
        """최적화 보정 백엔드에 직접 전달할 정제된 활성 데이터 셋 리스트를 반환합니다."""
        active_list = []
        for case in self.cases:
            if case["active"]:
                # JAX optimization format에 정렬
                # [{"times": jnp.array, "F_history": jnp.array, "target_diff": jnp.array}, ...]
                t = case["time"]
                e = case["strain"]
                s = case["stress"]
                
                # Deformation gradient F = diag(1+e, 1/sqrt(1+e), 1/sqrt(1+e))
                F_list = []
                for strain_val in e:
                    lam = 1.0 + strain_val
                    F_step = np.diag([lam, 1.0 / np.sqrt(lam), 1.0 / np.sqrt(lam)])
                    F_list.append(F_step)
                F_history = np.stack(F_list)
                
                active_list.append({
                    "name": case["name"],
                    "times": t,
                    "F_history": F_history,
                    "target_diff": s,  # 1축 응력 sigma_11 - sigma_22 (22=0이므로 sigma_11과 동치)
                    "weight": case["weight"],
                    "case_ref": case  # 결과값 업데이트를 위해 참조 바인딩
                })
        return active_list
