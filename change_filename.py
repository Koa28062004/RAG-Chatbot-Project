import os
import unicodedata
import re
import string

def is_vietnamese(text):
    # Check if any character is non-ASCII
    return any(ord(c) > 127 for c in text)

def clean_filename(text):
    try:
        # Normalize unicode
        nfkd = unicodedata.normalize('NFKD', text)
        ascii_text = "".join([c for c in nfkd if not unicodedata.combining(c)])
    except Exception:
        ascii_text = text

    # Remove control chars (ASCII < 32 or = 127)
    ascii_text = ''.join(
        c if (c in string.printable and ord(c) >= 32 and ord(c) != 127) else '_' 
        for c in ascii_text
    )

    # Replace spaces with underscores
    ascii_text = ascii_text.replace(' ', '_')

    # Remove any character not alphanumeric, dot, dash, underscore
    ascii_text = re.sub(r'[^A-Za-z0-9._-]', '', ascii_text)

    # Replace multiple underscores with single underscore
    ascii_text = re.sub(r'_+', '_', ascii_text)

    # Remove leading/trailing underscores
    ascii_text = ascii_text.strip('_')

    return ascii_text

def rename_vietnamese_files(root_folder):
    for dirpath, dirnames, filenames in os.walk(root_folder):
        for filename in filenames:
            if is_vietnamese(filename) or any(ord(c) < 32 or ord(c) == 127 for c in filename):
                new_filename = clean_filename(filename)
                old_path = os.path.join(dirpath, filename)
                new_path = os.path.join(dirpath, new_filename)
                try:
                    os.rename(old_path, new_path)
                    print(f"Renamed: {old_path} -> {new_path}")
                except Exception as e:
                    print(f"Failed to rename {old_path}: {e}")

# Example usage:
folder_path = "TONG_HOP_QC-TC_MEP_HIEN_HANH_md"
rename_vietnamese_files(folder_path)
