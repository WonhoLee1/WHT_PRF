"""
WHTOOLs MATCALIB 2026
--------------------------------------------------------------------------------
JAX vs Abaqus Validation Report Generator
Reads all generated _metrics.json files, compiles them into a markdown table,
and saves the artifact.

Usage:
    python generate_jax_report.py
--------------------------------------------------------------------------------
"""
import os
import glob
import json

def main():
    benchmark_dir = r"d:\PythonCodeStudy\WHT_PRF\benchmark_abaqus"
    os.chdir(benchmark_dir)
    
    json_files = glob.glob("*_metrics.json")
    
    results = []
    for jf in json_files:
        try:
            with open(jf, 'r') as f:
                data = json.load(f)
                results.append(data)
        except Exception as e:
            print(f"Error reading {jf}: {e}")
            
    # Sort by Stress R2 descending, handling Nones
    results.sort(key=lambda x: x.get('stress_r2') if x.get('stress_r2') is not None else -999, reverse=True)
    
    report_content = "# JAX WHT_PRF vs Abaqus Validation Metrics\n\n"
    report_content += "This table summarizes the $R^2$ and $RMSE$ scores between the JAX WHT_PRF prediction and Abaqus reference data for all executed models.\n\n"
    
    report_content += "| Model Name | Stress $R^2$ | Stress RMSE | Strain $R^2$ | Strain RMSE |\n"
    report_content += "|---|---|---|---|---|\n"
    
    for r in results:
        name = r.get('job_name', 'Unknown')
        str_r2 = f"{r.get('stress_r2'):.4f}" if r.get('stress_r2') is not None else "N/A"
        str_rmse = f"{r.get('stress_rmse'):.4f}" if r.get('stress_rmse') is not None else "N/A"
        stn_r2 = f"{r.get('strain_r2'):.4f}" if r.get('strain_r2') is not None else "N/A"
        stn_rmse = f"{r.get('strain_rmse'):.4f}" if r.get('strain_rmse') is not None else "N/A"
        
        report_content += f"| {name} | {str_r2} | {str_rmse} | {stn_r2} | {stn_rmse} |\n"
        
    artifact_dir = r"C:\Users\GOODMAN\.gemini\antigravity\brain\e8c44891-6b0e-4639-b300-0b7d95960f28"
    report_path = os.path.join(artifact_dir, "jax_validation_report.md")
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_content)
        
    print(f"Validation report generated at {report_path}")

if __name__ == "__main__":
    main()
