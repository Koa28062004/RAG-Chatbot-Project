import json
import re
import os
import glob
import unicodedata

# === 1️⃣ Load all image entries ===
with open("temp_process/image_doc.json", "r", encoding="utf-8") as f:
    all_image_entries = json.load(f)

def normalize_name(name):
    """
    - Remove Vietnamese accents
    - Replace spaces, dashes, dots with underscores
    - Collapse multiple underscores
    - Remove leading/trailing underscores
    - Lowercase for uniformity
    """
    name = unicodedata.normalize('NFD', name)
    name = ''.join(c for c in name if unicodedata.category(c) != 'Mn')
    name = re.sub(r"[.\s\-]+", "_", name)
    name = re.sub(r"_+", "_", name)
    name = name.strip("_")
    return name.lower()

# === 2️⃣ Build a map: normalized, unaccented PDF name => real PDF filename ===
pdf_folder = "TONG_HOP_QC-TC_MEP_HIEN_HANH_pdf"
pdf_files = glob.glob(os.path.join(pdf_folder, "**", "*.pdf"), recursive=True)
pdf_files += glob.glob(os.path.join(pdf_folder, "**", "*.PDF"), recursive=True)
pdf_name_map = {}

for pdf_path in pdf_files:
    pdf_base = os.path.splitext(os.path.basename(pdf_path))[0]
    norm_pdf = normalize_name(pdf_base)
    pdf_name_map[norm_pdf] = os.path.basename(pdf_path)

print(f"📑 Found {len(pdf_name_map)} PDF files.")

# === 3️⃣ Fuzzy matcher ===
def fuzzy_match(normalized_target, candidates):
    target_tokens = set(normalized_target.split("_"))
    best_match = None
    best_score = 0.0
    for candidate in candidates:
        candidate_tokens = set(candidate.split("_"))
        overlap = len(target_tokens & candidate_tokens)
        union = len(target_tokens | candidate_tokens)
        ratio = overlap / union if union > 0 else 0.0
        if ratio > best_score:
            best_score = ratio
            best_match = candidate
    if best_score >= 0.4:
        return best_match
    return None

# === 4️⃣ Main convert function ===
def convert_markdown_to_json(markdown_path, doc_name=None, filename=None):
    with open(markdown_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    if doc_name is None:
        raw_name = os.path.splitext(os.path.basename(markdown_path))[0]
        doc_name = normalize_name(raw_name)

    image_entries = [img for img in all_image_entries if img["id"].startswith(doc_name)]
    image_index = 0

    json_data = {
        "doc_name": doc_name,
        "filename": filename if filename else "",
        "sections": []
    }

    hierarchy_stack = []

    for line in lines:
        line = line.strip()
        if not line:
            continue

        heading_match = re.match(r'^(#+)\s+(.*)', line)
        if heading_match:
            hashes, title = heading_match.groups()
            level = len(hashes)

            if level == 1:
                section = {"section": title, "content": []}
                json_data["sections"].append(section)
                hierarchy_stack = [section]
            else:
                subsection = {"subsection": title, "content": []}
                while len(hierarchy_stack) >= level:
                    hierarchy_stack.pop()
                parent = hierarchy_stack[-1]
                parent["content"].append(subsection)
                hierarchy_stack.append(subsection)
            continue

        if not hierarchy_stack:
            continue

        # Check for markdown-style image
        image_md_match = re.match(r'!\[(.*?)\]\((.*?)\)', line.strip())
        if image_md_match:
            image_name = image_md_match.group(1).strip()
            image_url = image_md_match.group(2).strip()

            img_data = {
                "type": "image",
                "image_name": image_name,
                "image_markdown": image_url
            }

            hierarchy_stack[-1]["content"].append(img_data)
            continue

        text_data = {
            "type": "text",
            "data": line
        }
        hierarchy_stack[-1]["content"].append(text_data)

    return json_data

# === 5️⃣ Batch processing ===
input_folder = "TONG HOP QC-TC MEP HIEN HANH_Markdown_FIXED"
output_folder = "new-vn-data-json"
os.makedirs(output_folder, exist_ok=True)

markdown_files = [
    f for f in glob.glob(os.path.join(input_folder, "**", "*.md"), recursive=True)
    if "ENG" not in f
]

for md_file in markdown_files:
    raw_name = os.path.splitext(os.path.basename(md_file))[0]
    doc_name = normalize_name(raw_name)

    # Try exact match first
    pdf_filename = pdf_name_map.get(doc_name, "")

    # If not found, use fuzzy match
    if not pdf_filename:
        best = fuzzy_match(doc_name, pdf_name_map.keys())
        if best:
            pdf_filename = pdf_name_map[best]
            print(f"🔍 Fuzzy match: {raw_name} -> {pdf_filename} (score >= 40%)")
        else:
            print(f"⚠️ Warning: No PDF found for: {raw_name} (normalized: {doc_name})")

    json_output_path = os.path.join(output_folder, doc_name + ".json")
    json_data = convert_markdown_to_json(md_file, doc_name=doc_name, filename=pdf_filename)

    with open(json_output_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)

print(f"📂 Processed {len(markdown_files)} markdown files from {input_folder}")
