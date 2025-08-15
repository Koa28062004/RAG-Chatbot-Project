import os
from pathlib import Path

# Path to base folder containing the files
base_folder = Path("TONG_HOP_QC-TC_MEP_HIEN_HANH_docling/TONG HOP QC-TC MEP HIEN HANH")

# Read file paths from tmp.txt
with open("tmp.txt", "r", encoding="utf-8") as f:
    lines = [line.strip() for line in f if line.strip()]

# Loop through each path
for rel_path in lines:
    # Remove extension from the path
    no_ext_path = Path(rel_path).with_suffix("")  # removes .pdf
    target_dir = base_folder / no_ext_path.parent
    base_filename = no_ext_path.name

    # Search for any file with matching name (ignoring extension)
    if target_dir.exists():
        for file in target_dir.iterdir():
            if file.is_file() and file.stem == base_filename:
                print(f"Deleting: {file}")
                file.unlink()
