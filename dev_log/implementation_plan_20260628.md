# Abaqus Verification Benchmark Suite

This document outlines the plan to create a verification benchmark suite for the JAX-based WHT_PRF model using Abaqus `.inp` files from the SIMULIA documentation.

## Goal Description
1. Create a `benchmark_abaqus` folder.
2. Develop a generalized `abaqus_reader.py` module to parse single-element Abaqus `.inp` files. It will extract material properties (Hyperelastic and PRF visco network parameters) and simulation steps/boundary conditions.
3. Copy the relevant `.inp` files (referenced in `simaver-c-nonlinviscohyper.htm`) into the `benchmark_abaqus` folder.
4. For each example, automatically generate a `.py` script that uses the `abaqus_reader` to set up the JAX-based WHT_PRF model, run the simulation corresponding to the `.inp` steps, and save a result graph as a PNG.
5. If an `.inp` file has more than one element or is not a hexahedral element (e.g., C3D8), the reader will issue a warning but attempt to process it as a single hexahedral element test.

> [!WARNING]
> **Abaqus Reference Data**: The `.inp` files themselves do not contain the output data (stress/strain curves). Unless we also have `.dat` or `.odb` files with the reference results, the generated PNGs will only show the results predicted by the JAX WHT_PRF model.

## Open Questions

> [!IMPORTANT]
> 1. **Scope of Scripts**: There are over 50 `.inp` files referenced. Should I generate individual `.py` files for **all** of them, or start with a representative subset (e.g., covering different hyperelastic and creep laws)?
> 2. **Reference Comparison**: Do you have Abaqus output files (like `.dat` or `.rpt` files) containing the reference results to plot against? If not, the "comparison graphs" will currently only plot the JAX PRF results. 
> 3. **Input Data Extraction**: `abaqus_reader.py` will read the hyperelastic constants and visco network definitions. Since JAX PRF takes specific parameters, the reader will map Abaqus keywords (e.g., NEOHOOK, OGDEN, STRAIN, HYPERB) to our Python classes. Is this mapping expected to cover all variants (e.g., Ogden, Arruda-Boyce, Van der Waals)?

## Proposed Changes

### `benchmark_abaqus/`
A new directory to contain all benchmark files.

### `benchmark_abaqus/abaqus_reader.py`
A generalized parser for Abaqus `.inp` files.
- Extracts `*HYPERELASTIC` properties.
- Extracts `*VISCOELASTIC, NONLINEAR` network properties.
- Extracts `*STEP`, `*STATIC`, `*VISCO` procedures and loading conditions (`*DLOAD`, `*BOUNDARY`).
- Identifies if the mesh is a single C3D8 element. If not, prints a warning message.

### `benchmark_abaqus/generate_benchmarks.py` (or individual `[name].py`)
A script to read each copied `.inp`, configure the JAX PRF model, run the simulation, and generate the PNG plots. It will generate individual Python scripts for each INP file as requested.

## Verification Plan
- Run one of the generated Python scripts (e.g., for `viscnet_c3d8_nh_n3.inp`) to verify that `abaqus_reader.py` correctly parses the material and loading step.
- Verify that the JAX WHT_PRF model successfully completes the simulation and generates a PNG plot.
