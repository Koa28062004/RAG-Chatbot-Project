# RAG-Chatbot

## Overview
RAG-Chatbot is a Retrieval-Augmented Generation (RAG) chatbot designed to process PDF documents, extract relevant data (text, images, and tables), and generate responses based on the extracted information. It leverages tools like ChromaDB for database management and Streamlit for deployment.

## Features
- Converts PDF files to Markdown.
- Extracts images, captions, and tables from PDFs.
- Loads extracted data into ChromaDB for efficient querying.
- Generates responses based on the processed data.

## Prerequisites
- Python 3.9 or higher
- A valid Gemini API key (add it to a `.env` file as shown in `.env.example`).

## Installation

1. **Set up the virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   
   ```bash
   python ./setup/setup.py
   ```

3. **Configure environment variables**:
   - Copy `.env.example` to `.env`:
     ```bash
     cp .env.example .env
     ```
   - Add your Gemini API key to the `.env` file.

## Usage

### 1. Process PDF Data
Run the `convert_pdf_to_md.py` script to process PDF files and extract data:
```bash
python convert_pdf_to_md.py
```
This script will:
- Convert PDFs in the `data/` folder to Markdown in the `data-md/` folder.

### 2. Fix the headings between markdown files and pdf files
Run the `utils/fix_headings.py` script to fix heading:
```bash
python utils/fix_headings.py
```

### 3. Divide the subsections between level 1, 2,...
Run the `utils/divide_subsection.py` script to divide subsections:
```bash
python utils/divide_subsection.py
```

### 4. Split chunks and embedding
Run the `main.py` script to divide subsections:
```bash
python utils/divide_subsection.py
```

### 5. Deploy the Chatbot
Run the Streamlit app:
```bash
python3 -u -m uvicorn app:app --host 0.0.0.0 --port 3000
```
Access the chatbot in your browser at `http://localhost:3000`.