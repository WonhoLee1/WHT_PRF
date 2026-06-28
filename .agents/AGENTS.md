# 개발 이력 규칙 (Development Logging Rules)
* **[필수]** dev_log 폴더에 계획, 구현 내용을 매 산출마다 기록합니다.
* **[필수]** 개발이력에 대한 내용은 dev_log 폴더 내에 새로운 md 파일(예: `development_history.md`)을 만들어서 함께 합니다.
* 매일 개발이 종료되거나 주요 마일스톤 돌파 시, 해당 시점의 `./dev_log/` 폴더 내에 계획(`implementation_plan_YYYYMMDD.md`) 및 태스크 목록(`task_YYYYMMDD.md`) 스냅샷을 백업합니다.
* 중대한 회귀(Regression) 오류나 설계 변경 발생 시, 해당 기록을 기반으로 이력을 추적합니다.

# 코딩 표준 및 브랜딩 (Coding Standards & Branding)
* **[필수]** 생성하는 모든 파이썬 소스 코드 파일 최상단에 `WHTOOLs MATCALIB 2026` 주석을 반드시 삽입합니다.
* **[필수]** 생성하는 함수 및 클래스에는 파라미터, 반환값, 용도를 명시한 Docstring 주석을 반드시 삽입합니다.
* **[필수]** 실행 가능한 스크립트의 경우, 파일 하단에 활용 예제(Usage)를 포함한 `if __name__ == "__main__":` 블록을 반드시 삽입합니다.

# 인코딩 및 파일 입출력 규칙 (Encoding & File I/O Rules)
* **[필수]** Windows PowerShell에서 `Add-Content`, `Out-File` 혹은 `>`, `>>` 리다이렉션을 사용하여 텍스트 파일(md 등)을 수정하거나 덧붙이면 UTF-16 인코딩 강제 변환 및 혼용으로 인해 문서 전체가 심각하게 깨집니다.
* **[필수]** 기존 파일에 텍스트를 추가하거나 인코딩을 다뤄야 할 때는 **절대 PowerShell 커맨드를 사용하지 말고, 반드시 Python 스크립트(`open(..., 'a', encoding='utf-8')`) 혹은 전용 편집 툴(multi_replace_file_content 등)을 사용**하여 수정합니다.
* **[필수]** 모든 새 파일 및 업데이트 파일은 반드시 BOM 없는 UTF-8 (UTF-8 without BOM) 포맷으로 저장 및 취급합니다.
