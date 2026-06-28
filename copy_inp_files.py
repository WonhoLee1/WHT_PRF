import os
import re
import shutil

html_path = r"C:\SIMULIA\Documentation\2024LE\English\SIMACAEVERRefMap\simaver-c-nonlinviscohyper.htm"
inp_dir = r"C:\SIMULIA\Documentation\2024LE\English\SIMAINPRefResources"
dest_dir = r"d:\PythonCodeStudy\WHT_PRF\benchmark_abaqus"

if not os.path.exists(dest_dir):
    os.makedirs(dest_dir)

with open(html_path, "r", encoding="utf-8", errors="ignore") as f:
    content = f.read()

# Pattern to find all .inp files referenced in the HTML
pattern = re.compile(r'href\s*=\s*"[^"]+/(.*?\.inp)"')
matches = pattern.findall(content)

# Use a set to avoid duplicates
inp_files = sorted(list(set(matches)))

print(f"Found {len(inp_files)} unique INP files.")

copied_count = 0
for filename in inp_files:
    src_path = os.path.join(inp_dir, filename)
    dest_path = os.path.join(dest_dir, filename)
    if os.path.exists(src_path):
        shutil.copy2(src_path, dest_path)
        copied_count += 1
    else:
        print(f"Warning: File not found {src_path}")

print(f"Successfully copied {copied_count} files to {dest_dir}.")
