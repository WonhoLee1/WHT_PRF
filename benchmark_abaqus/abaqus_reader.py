"""
WHTOOLs MATCALIB 2026
--------------------------------------------------------------------------------
Abaqus INP File Reader Module
Parses Hyperelastic and Viscoelastic properties, step, and boundary conditions
from Abaqus input files.

Usage:
    from abaqus_reader import AbaqusINPReader
    reader = AbaqusINPReader("path/to/file.inp")
    print(reader.hyperelastic)
--------------------------------------------------------------------------------
"""
import os
import re

class AbaqusINPReader:
    def __init__(self, filepath):
        self.filepath = filepath
        self.element_type = None
        self.element_count = 0
        self.hyperelastic = {'model': 'UNKNOWN', 'params': []}
        self.mullins = None
        self.visco_networks = []
        self.steps = []
        self.parse()

    def parse(self):
        try:
            with open(self.filepath, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
        except Exception as e:
            print(f"Error reading file {self.filepath}: {e}")
            return
            
        current_section = None
        current_step = None
        
        for i, line in enumerate(lines):
            original_line = line.strip()
            line = original_line.upper()
            if not line or line.startswith('**'):
                continue
            
            if line.startswith('*ELEMENT') and not line.startswith('*ELEMENT OUTPUT'):
                m = re.search(r'TYPE=([A-Z0-9]+)', line)
                if m:
                    self.element_type = m.group(1)
                current_section = 'ELEMENT'
            elif current_section == 'ELEMENT' and line.startswith('*'):
                current_section = None
            elif current_section == 'ELEMENT':
                self.element_count += 1
                
            elif line.startswith('*HYPERELASTIC'):
                parts = line.split(',')
                if len(parts) > 1:
                    self.hyperelastic['model'] = parts[1].strip()
                current_section = 'HYPERELASTIC'
            elif current_section == 'HYPERELASTIC' and line.startswith('*'):
                current_section = None
            elif current_section == 'HYPERELASTIC':
                vals = [float(v) for v in line.split(',') if v.strip()]
                self.hyperelastic['params'] = vals
                current_section = None
                
            elif line.startswith('*MULLINS EFFECT'):
                current_section = 'MULLINS'
                self.mullins = {'params': []}
            elif current_section == 'MULLINS' and line.startswith('*'):
                current_section = None
            elif current_section == 'MULLINS':
                vals = [float(v) for v in line.split(',') if v.strip()]
                self.mullins['params'] = vals
                current_section = None
            elif line.startswith('*VISCOELASTIC'):
                network = {}
                parts = line.split(',')
                for p in parts:
                    if '=' in p:
                        k, v = p.split('=')
                        network[k.strip()] = v.strip()
                current_section = 'VISCOELASTIC'
                network['params'] = []
                self.visco_networks.append(network)
            elif current_section == 'VISCOELASTIC' and line.startswith('*'):
                current_section = None
            elif current_section == 'VISCOELASTIC':
                vals = [float(v) for v in line.split(',') if v.strip()]
                self.visco_networks[-1]['params'] = vals
                current_section = None
                
            elif line.startswith('*STEP'):
                current_step = {'type': 'UNKNOWN', 'duration': 1.0, 'load': []}
                self.steps.append(current_step)
            elif line.startswith('*END STEP'):
                current_step = None
            
            elif current_step is not None:
                if line.startswith('*STATIC'):
                    current_step['type'] = 'STATIC'
                    current_section = 'STATIC_PARAMS'
                elif line.startswith('*VISCO'):
                    current_step['type'] = 'VISCO'
                    current_section = 'VISCO_PARAMS'
                elif current_section in ['STATIC_PARAMS', 'VISCO_PARAMS'] and not line.startswith('*'):
                    vals = [float(v) for v in line.split(',') if v.strip()]
                    if len(vals) >= 2:
                        current_step['duration'] = vals[1]
                    elif len(vals) == 1:
                        current_step['duration'] = vals[0]
                    current_section = None
                elif line.startswith('*DLOAD'):
                    current_section = 'DLOAD'
                elif current_section == 'DLOAD' and not line.startswith('*'):
                    parts = [p.strip() for p in original_line.split(',')]
                    if len(parts) >= 3:
                        current_step['load'].append({
                            'type': 'DLOAD',
                            'elset': parts[0],
                            'face': parts[1],
                            'magnitude': float(parts[2])
                        })
                elif line.startswith('*BOUNDARY'):
                    current_section = 'BOUNDARY'
                elif current_section == 'BOUNDARY' and not line.startswith('*'):
                    parts = [p.strip() for p in original_line.split(',')]
                    current_step['load'].append({'type': 'BOUNDARY', 'details': parts})
                elif line.startswith('*'):
                    current_section = None
                    
    def check_compatibility(self):
        warnings = []
        if self.element_count > 1:
            warnings.append(f"Model has {self.element_count} elements. Expecting a single element test.")
        if self.element_type and not self.element_type.startswith('C3D8'):
            warnings.append(f"Element type {self.element_type} is not a standard 3D hexahedral (C3D8). We will treat it as a hexahedral test.")
        return warnings

    def print_summary(self):
        print(f"--- Summary for {os.path.basename(self.filepath)} ---")
        warnings = self.check_compatibility()
        for w in warnings:
            print(f"[WARNING] {w}")
        print(f"Elements: {self.element_count} x {self.element_type}")
        print(f"Hyperelastic: {self.hyperelastic['model']}, params={self.hyperelastic.get('params', [])}")
        for i, net in enumerate(self.visco_networks):
            print(f"Visco Network {i+1}: LAW={net.get('LAW')}, SRATIO={net.get('SRATIO')}, params={net.get('params', [])}")
        for i, step in enumerate(self.steps):
            print(f"Step {i+1}: {step['type']} (Duration: {step['duration']})")
            for ld in step['load']:
                print(f"  Load: {ld}")
        print("-" * 40)

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        reader = AbaqusINPReader(sys.argv[1])
        reader.print_summary()
