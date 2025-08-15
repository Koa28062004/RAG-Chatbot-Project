import os
import subprocess

def convert_docs_to_pdf(folder_path):
    for filename in os.listdir(folder_path):
        if filename.lower().endswith((".doc", ".docx")):
            input_path = os.path.join(folder_path, filename)
            try:
                subprocess.run([
                    "libreoffice",
                    "--headless",
                    "--convert-to", "pdf",
                    "--outdir", folder_path,
                    input_path
                ], check=True)
                print(f"Converted: {filename}")
            except subprocess.CalledProcessError as e:
                print(f"Failed to convert {filename}: {e}")

# Example usage
folder = "check"
convert_docs_to_pdf(folder)
