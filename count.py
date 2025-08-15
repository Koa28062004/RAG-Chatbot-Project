import os

def count_files_in_folder(folder_path):
    total_files = 0
    for root, dirs, files in os.walk(folder_path):
        total_files += len(files)
    return total_files

# Example usage
folder_to_count = "TONG HOP QC-TC MEP HIEN HANH_Markdown_FIXED"
num_files = count_files_in_folder(folder_to_count)
print(f"Total files in '{folder_to_count}': {num_files}")
