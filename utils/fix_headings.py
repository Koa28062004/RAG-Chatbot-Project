import re
import os

def fix_markdown_headings_single_file(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # Step 1: Force all headings to #
    lines = [re.sub(r'^#+', '#', line) for line in lines]

    new_lines = []
    for line in lines:
        original_line = line  # keep for debugging if needed

        # Fix multiple list dashes:
        # 1) Pattern "- -" at start becomes "- "
        line = re.sub(r'^(-\s+-\s+)', '- ', line)

        line = re.sub(r'-\s*\++\s*', '- ', line)

        # 2) Special case: "- -(..." should become "- (...)"
        line = re.sub(r'^-\s+-\(\s*', '- (', line)

        # Match numeric headings: 1.2.3 Title
        m_num = re.match(r'^#\s+(\d+(\.\d+)*)(.*)$', line.strip())
        # Match letter + numeric: C.1 or C.1.2.3 Title
        m_letter = re.match(r'^#\s+([A-Z]\.(\d+(\.\d+)*))(.*)$', line.strip())

        if m_num:
            number, _, title = m_num.groups()
            level = number.count('.') + 1
            hashes = '#' * level
            new_line = f"{hashes} {number}{title}\n"
            new_lines.append(new_line)
        elif m_letter:
            full_number, number_part, _, title = m_letter.groups()
            level = full_number.count('.') + 1
            hashes = '#' * level
            new_line = f"{hashes} {full_number}{title}\n"
            new_lines.append(new_line)
        else:
            new_lines.append(line)

    with open(output_file, 'w', encoding='utf-8') as f:
        f.writelines(new_lines)


def process_folder_tree(input_root, output_root):
    for root, dirs, files in os.walk(input_root):
        rel_path = os.path.relpath(root, input_root)
        output_dir = os.path.join(output_root, rel_path)
        os.makedirs(output_dir, exist_ok=True)

        for file in files:
            if file.endswith('.md'):
                input_file = os.path.join(root, file)
                output_file = os.path.join(output_dir, file)
                fix_markdown_headings_single_file(input_file, output_file)
                print(f"✅ Processed: {input_file} → {output_file}")


# === Example usage ===
input_root = 'TONG_HOP_QC-TC_MEP_HIEN_HANH_docling'
output_root = 'TONG HOP QC-TC MEP HIEN HANH_Markdown_FIXED'

process_folder_tree(input_root, output_root)
