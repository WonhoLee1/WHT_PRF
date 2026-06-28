import os
import re
import numpy as np

def parse_radioss_t01(t01_file):
    """
    Parses an OpenRadioss T01 (Time History) ASCII file.
    Returns a dictionary of numpy arrays.
    """
    if not os.path.exists(t01_file):
        raise FileNotFoundError(f"T01 file not found: {t01_file}")
        
    data = {}
    channel_names = []
    
    with open(t01_file, 'r', encoding='utf-8', errors='replace') as f:
        lines = f.readlines()
        
    # T01 format has headers and then data. 
    # Usually the first few lines specify variables.
    
    in_data = False
    data_lines = []
    
    for i, line in enumerate(lines):
        if not in_data:
            # Look for channel names or the end of header
            if line.startswith('/'):
                pass # Keyword
            elif re.match(r'^\s*[-.0-9E+]+', line) and len(line.split()) > 1:
                # Looks like data
                in_data = True
                data_lines.append(line)
        else:
            data_lines.append(line)
            
    if not data_lines:
        return data
        
    # Convert data lines to numpy array
    # Radioss T01 might wrap long lines or just have very long lines.
    # Usually it's multiple columns.
    
    # Try parsing flat values
    all_vals = []
    for line in data_lines:
        # Some Fortran outputs lack 'E' before exponent, e.g. 1.234-100 instead of 1.234E-100
        # Replace occurrences like "1.234-100" with "1.234E-100" if necessary, though Radioss uses E.
        line = re.sub(r'(?<=\d)([-+])(?=\d{2,3}\b)', r'E\1', line)
        vals = [float(x) for x in line.split() if x.strip()]
        all_vals.extend(vals)
        
    # Since we don't know the number of channels easily if it wraps,
    # let's try to infer from the number of columns of the first data line (assuming no wrap)
    # Actually, T01 usually doesn't wrap or if it does, it's consistent.
    first_line_cols = len([x for x in data_lines[0].split() if x.strip()])
    # Assuming standard rectangular matrix
    matrix = []
    for line in data_lines:
        matrix.append([float(x) for x in line.split() if x.strip()])
        
    # Pad or reshape if needed. Usually it is safe to just use matrix if no wrapping.
    arr = np.array(matrix)
    
    if len(arr.shape) == 2:
        data['time'] = arr[:, 0]
        # We need to identify what other columns are. Usually:
        # Col 0: Time
        # Following cols depend on /TH definitions.
        # For /TH/NODE, maybe Displacements, Velocities, Forces.
        # We will just return the full array and let the caller figure it out
        data['raw_matrix'] = arr
        
    return data
