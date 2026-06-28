"""
WHTOOLs MATCALIB 2026
--------------------------------------------------------------------------------
Markdown Report Generator
Compiles all generated Abaqus comparison PNGs into a single Markdown report.

Usage:
    python generate_report.py
--------------------------------------------------------------------------------
"""
import os
import glob
import shutil

def main():
    benchmark_dir = r"d:\PythonCodeStudy\WHT_PRF\benchmark_abaqus"
    os.chdir(benchmark_dir)
    
    png_files = glob.glob("*_graphs.png")
    png_files.sort()
    
    # We will write this directly to the artifact directory.
    # We first copy all pngs to the artifact directory so they can be viewed.
    artifact_dir = r"C:\Users\GOODMAN\.gemini\antigravity\brain\e8c44891-6b0e-4639-b300-0b7d95960f28"
    
    report_content = "# Abaqus Result Graphs\n\n"
    report_content += "The following are the extracted Time-Strain, Time-Stress, and Strain-Stress graphs for all 50+ Abaqus models.\n\n"
    
    for png in png_files:
        src_path = os.path.join(benchmark_dir, png)
        dest_path = os.path.join(artifact_dir, png)
        
        # Copy to artifact dir so markdown can render it natively
        try:
            shutil.copy2(src_path, dest_path)
            report_content += f"## {png.replace('_graphs.png', '')}\n"
            report_content += f"![{png}](file:///{dest_path.replace(chr(92), '/')})\n\n"
        except Exception as e:
            print(f"Error copying {png}: {e}")
            
    report_path = os.path.join(artifact_dir, "abaqus_50_results_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_content)
        
    print(f"Report generated at {report_path}")

if __name__ == "__main__":
    main()
