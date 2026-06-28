# Representative Models & Abaqus Execution Plan

This document outlines the steps to select representative models, execute them in Abaqus within a completely isolated network environment, and extract the reference results for direct comparison with JAX WHT_PRF.

## Goal Description
1. Identify and label representative models (e.g., Neo-Hookean with Power Law Creep) with a `REP_` prefix.
2. Configure Windows Firewall rules to strictly block all inbound and outbound network traffic for Abaqus executables (`standard.exe`, `ABQcaeK.exe`, `abaqus.exe`).
3. Run the isolated Abaqus jobs on the representative models to generate `.odb` files.
4. Develop a utility script (`extract_odb.py`) to automatically extract the time, stress (S11), and strain (LE11/E11) data from the `.odb` files into text files.
5. Update the JAX benchmarking scripts (`REP_run_*.py`) to read these text files and plot both the Abaqus reference data and the JAX PRF simulation results on the same graph for validation.

## Open Questions

> [!IMPORTANT]
> 1. **Representative Selection**: I will select a few key examples (e.g. `viscnet_c3d8_nh_n2.inp`, `viscnet_c3d8_nh_n3.inp`) as the "representative" models and prefix them with `REP_`. Do you have specific INP files you prefer to serve as the representatives?
> 2. **Network Blocking Mechanism**: I will use PowerShell's `New-NetFirewallRule` to block `standard.exe`, `explicit.exe`, and `ABQcaeK.exe` within the `c:\SIMULIA\CAE\2024LE\win_b64\code\bin\` directory. Are there any other specific Abaqus-related components you want to ensure are blocked?

## Proposed Changes

### 1. `setup_firewall.ps1`
A script that registers Windows Firewall rules to block inbound and outbound traffic for all Abaqus binaries before any simulation starts.

### 2. `extract_odb.py`
A python script designed to be run via `abaqus python extract_odb.py <odb_filename>`. It will access the Abaqus ODB API, locate the single C3D8 element's integration point, and dump the Time vs Stress (S11) and Time vs Strain (E11) history to a `.txt` file.

### 3. Updated `generate_benchmarks.py`
Will be modified to:
- Add a `REP_` prefix to the generated `.py` files for the chosen representative INPs.
- Alter the Matplotlib plotting code in the representative `.py` files to load the extracted `*_abaqus_results.txt` and plot it against the JAX prediction.

## Verification Plan
- Execute the firewall script and confirm the rules are active in Windows Firewall.
- Run one Abaqus job (`abaqus job=viscnet_c3d8_nh_n3 interactive`) and ensure it completes without network access.
- Run the extraction script and verify the text file contains valid numeric data.
- Run the `REP_` JAX python script and confirm the final PNG displays both curves.
