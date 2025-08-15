import os
import re
import fitz  # PyMuPDF
from docling.document_converter import DocumentConverter
import re
import json
from tqdm import tqdm

def extract_caption_text(raw_caption: str) -> str:
    """
    Extract the part after ':' or '-' in captions like 'Bảng 2.2: ...' or 'Hình 3 - ...'.
    """
    match = re.split(r"[:\-\.]", raw_caption, maxsplit=1)
    if len(match) == 2:
        return match[1].strip()
    return raw_caption.strip()

def clean_table_separators(markdown_text: str) -> str:
    lines = markdown_text.splitlines()
    cleaned_lines = []
    pending_caption = None
    inside_table = False
    first_separator_found = False
    skip_next_separator = False

    for i, line in enumerate(lines):
        stripped = line.strip()

        if skip_next_separator:
            # Skip the separator line following the bad "Bảng ..." row
            if re.match(r"^\|?(\s*:?-+:?\s*\|)+\s*$", stripped):
                skip_next_separator = False
                continue

        # Detect mis-OCRed table title inside a table (e.g. "| Bảng 5 ... | Bảng 5 ... |")
        if (
            stripped.startswith("|") and stripped.endswith("|") and
            any(re.search(r"\bB(ả|a)ng\b", cell.strip(), re.IGNORECASE) for cell in stripped.split("|") if cell.strip())
        ):
            # Extract most informative cell as caption
            cells = [cell.strip() for cell in stripped.split("|") if cell.strip()]
            most_common_caption = max(cells, key=len) if cells else None
            if most_common_caption:
                pending_caption = f"{most_common_caption}"
                skip_next_separator = True  # Also skip the separator row after this
            continue

        # Detect a markdown table separator row
        if re.match(r"^\|?(\s*:?-+:?\s*\|)+\s*$", stripped):
            if inside_table:
                if not first_separator_found:
                    cleaned_lines.append(line)
                    first_separator_found = True
                # Extra separators inside table are skipped
            else:
                cleaned_lines.append(line)
            continue

        # If table row, mark as inside a table
        if stripped.startswith("|") and stripped.endswith("|"):
            if pending_caption:
                cleaned_lines.append(pending_caption)
                pending_caption = None
            inside_table = True
        else:
            inside_table = False
            first_separator_found = False

        cleaned_lines.append(line)

    return "\n".join(cleaned_lines)

def convert_pdf_to_markdown(pdf_folder: str, md_output_folder: str):
    os.makedirs(md_output_folder, exist_ok=True)

    for filename in tqdm(os.listdir(pdf_folder), desc="Converting PDFs to markdown"):
        if filename.lower().endswith(".pdf"):
            pdf_path = os.path.join(pdf_folder, filename)
            print(f"📄 Converting: {pdf_path}")

            try:
                converter = DocumentConverter()
                result = converter.convert(pdf_path)
                markdown_text = result.document.export_to_markdown()

                # Remove empty lines (optional)
                lines = markdown_text.splitlines()
                non_empty_lines = [line for line in lines if line.strip() != ""]
                cleaned_markdown = "\n".join(non_empty_lines)

                # Clean repeated table separators caused by page breaks
                cleaned_markdown = clean_table_separators(cleaned_markdown)

                md_filename = os.path.splitext(filename)[0] + ".md"
                md_path = os.path.join(md_output_folder, md_filename)

                with open(md_path, "w", encoding="utf-8") as f:
                    f.write(cleaned_markdown)

                print(f"✅ Saved cleaned markdown: {md_path}")
            except Exception as e:
                print(f"❌ Error processing {filename}: {e}")

def extract_images_and_caption_flexible(pdf_path, md_folder, image_save_dir):
    os.makedirs(image_save_dir, exist_ok=True)
    doc = fitz.open(pdf_path)
    image_files = []

    base_name = os.path.splitext(os.path.basename(pdf_path))[0]
    md_path = os.path.join(md_folder, base_name + ".md")

    # Extract images from PDF
    for page_num in range(len(doc)):
        page = doc[page_num]
        for img_index, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            ext = base_image["ext"]

            filename = f"{base_name}_p{page_num}_{xref}.{ext}"
            image_path = os.path.join(image_save_dir, filename)

            with open(image_path, "wb") as f:
                f.write(image_bytes)

            image_files.append(image_path)

    # Parse markdown captions
    if not os.path.exists(md_path):
        print(f"⚠️ Markdown not found for {pdf_path}")
        return []

    with open(md_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f.readlines()]

    captions = [line for line in lines if line.startswith("Hình")]

    # Build list of dicts for each image+caption
    image_docs = []
    for i, image_path in enumerate(image_files):
        raw_caption = captions[i] if i < len(captions) else "No Caption"
        caption = extract_caption_text(raw_caption)
        if not caption:
            caption = f"No Caption"
        image_id = os.path.splitext(os.path.basename(image_path))[0]
        image_docs.append({
            "id": image_id,
            "content": caption,
            "url": image_path
        })

    return image_docs

def extract_tables_from_markdown(md_folder, output_folder, table_doc_path):
    os.makedirs(output_folder, exist_ok=True)
    print("\n📊 Extracting tables from markdown...")

    all_tables_info = []  # To collect caption + url info for all tables

    for filename in os.listdir(md_folder):
        if filename.lower().endswith(".md"):
            md_path = os.path.join(md_folder, filename)
            print(f"Processing tables in: {md_path}")

            with open(md_path, "r", encoding="utf-8") as f:
                lines = [line.rstrip() for line in f.readlines()]

            tables = []
            current_table_lines = []
            current_title = None
            inside_table = False

            for i, line in enumerate(lines):
                # If line starts with Bảng - capture title before the table
                if re.match(r"^\s*#*\s*(B[Ả]N[G]{1,2})\b", line, re.IGNORECASE) or re.match(r"^\s*#*\s*bảng\b", line, re.IGNORECASE):
                    # If we were inside a table, save previous
                    if inside_table and current_table_lines:
                        tables.append((current_title, current_table_lines))
                        current_table_lines = []
                        inside_table = False

                    current_title = line.strip()

                # Check if this line looks like a markdown table row (starts with | and has | inside)
                if line.startswith("|") and "|" in line:
                    current_table_lines.append(line)
                    inside_table = True
                else:
                    # If line breaks table block
                    if inside_table and current_table_lines:
                        tables.append((current_title, current_table_lines))
                        current_table_lines = []
                        inside_table = False
                        current_title = None

            # Catch any leftover table at end of file
            if inside_table and current_table_lines:
                tables.append((current_title, current_table_lines))

            # Save each table to separate md file
            for idx, (title, table_lines) in enumerate(tables):
                safe_title = title if title else f"table_{idx+1}"
                safe_title = re.sub(r'[^\w\s-]', '', safe_title).replace(' ', '_')
                safe_title = safe_title[:50]  # Limit title length to 50 characters
                output_file = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}_{safe_title}.md")

                with open(output_file, "w", encoding="utf-8") as out_f:
                    if title:
                        out_f.write(title + "\n\n")
                    out_f.write("\n".join(table_lines))

                print(f"Saved table to {output_file}")

                # Save caption and relative path (URL)
                # Here URL is relative path from output_folder root or absolute path - adapt as needed
                url_path = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}_{safe_title}.md")
                all_tables_info.append({
                    "id": f"{os.path.splitext(filename)[0]}_{safe_title}",
                    "content": extract_caption_text(title) if title else "No Caption",
                    "url": url_path.replace("\\", "/")  # For consistent URL format on Windows
                })

    with open(table_doc_path, "w", encoding="utf-8") as json_f:
        json.dump(all_tables_info, json_f, ensure_ascii=False, indent=2)

    print(f"\nSaved all table metadata to {table_doc_path}")