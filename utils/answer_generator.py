import re
import difflib
import time
import unicodedata
import os
import json
import numpy as np

from collections import defaultdict
from typing import Dict, Tuple, List
from config import Config
from utils.search_query import QueryProcessor
from utils.embedding import BM25EmbeddingFunction


class AnswerGenerator:
    def __init__(self):
        self.config = Config()
        self.model = self.config.get_model_used()
        self.query_processor = QueryProcessor()
        self.FALLBACK_TRIGGERS = [
            "tôi chưa tìm thấy thông tin",
            "tôi không tìm thấy thông tin",
            "rất tiếc, hiện tại tôi chưa",
            "bạn vui lòng liên hệ chuyên viên",
            "tôi xin lỗi",
            "vui lòng liên hệ chuyên viên"
        ]

    def generate_openai_answer(self, prompt: str) -> str:
        start = time.time()
        response = self.model.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=2048
        )
        print(f"⏱️ OpenAI generation time: {time.time() - start:.2f}s")
        return response.choices[0].message.content.strip()

    def generate_gemini_answer(self, prompt: str):
        response = self.model.generate_content(prompt)
        return response.text

    def generate_model_answer(self, prompt: str) -> str:
        if self.config.MODEL_USED == "gemini":
            return self.generate_gemini_answer(prompt)
        elif self.config.MODEL_USED == "openai":
            return self.generate_openai_answer(prompt)
        else:
            raise ValueError(f"Unsupported model: {self.config.MODEL_USED}")

    def normalize_viet(self, query: str) -> str:
        prompt = f"""
        Bạn là một trợ lý chuyên xử lý ngôn ngữ chuyên ngành phòng cháy chữa cháy (PCCC).
        Hãy viết lại truy vấn dưới đây thành một câu hỏi đầy đủ, rõ ràng, sử dụng ngôn ngữ kỹ thuật chuẩn như trong các tài liệu QCVN hoặc TCVN.
        Mục tiêu:
        - Giữ nguyên từ khóa chính và nội dung gốc của truy vấn.
        - Bổ sung đầy đủ bối cảnh, tình huống hoặc điều kiện nếu có thể, để giúp truy vấn dễ được tìm thấy trong tài liệu kỹ thuật.
        - Tránh viết tắt, từ lóng hoặc ngôn ngữ địa phương.
        Chỉ trả về truy vấn đã được viết lại, không cần giải thích.

        Truy vấn gốc: "{query}"

        Truy vấn đã mở rộng và chuẩn hóa:
        """
        response = self.generate_model_answer(prompt)
        return response

    def normalize_eng(self, query: str) -> str:
        prompt = f"""
        You are a professional assistant specializing in fire prevention and fighting (PCCC) technical language.
        Rewrite the following query clearly, avoiding abbreviations or local dialects.
        The goal is to make this query easily understandable and searchable in standard technical documents like QCVN, TCVN.
        Only return the standardized question, no additional explanations.

        Original query: "{query}"

        Standardized query (using clear technical language):
        """
        response = self.generate_model_answer(prompt)
        return response

    def make_rag_prompt_viet(self, query: str, context_block: str) -> str:
        prompt = f"""
    Bạn là một trợ lý ảo chuyên nghiệp, có nhiệm vụ trả lời câu hỏi của khách hàng dựa trên nội dung tài liệu kỹ thuật hoặc văn bản pháp luật được cung cấp bên dưới.

    **Hướng dẫn trả lời:**
    1. **Phân tích kỹ câu hỏi và xác định các từ khóa quan trọng.** So sánh trực tiếp các từ khóa này với tiêu đề phần/chương trong tài liệu để tìm phần phù hợp nhất. Ví dụ, **hút khói** khác với **ống thông gió** — cần phân biệt rõ.
    2. Hãy chọn các đoạn văn liên quan đến câu hỏi, chú ý đến các từ khoá trong câu hỏi và tiêu đề phần/chương trong tài liệu. Chọn toàn bộ các đoạn văn có chứa từ khóa chính liên quan đến câu hỏi.
    3. Phải rà soát toàn bộ nội dung để tìm tất cả tiêu đề phần nào có chứa thông tin liên quan đến từ khóa trong câu hỏi. Không được chỉ dừng lại ở phần đầu tiên tìm thấy. Nếu nhiều phần có liên quan, phải chọn tất cả.

    🔴 4. **Bắt buộc giữ nguyên toàn bộ các nội dung có cấu trúc phân mục như `a)`, `b)`, `1.`, `2.`, hoặc mã số như `A.2.29.1`, `6.11.4`... Nếu một mục như `A.2.29` được chọn, phải bao gồm toàn bộ các tiểu mục `A.2.29.x` bên dưới. Không được bỏ sót hoặc rút gọn.**

    🔴 5. **Không được rút gọn, tóm tắt, gom nhóm hoặc diễn giải lại nội dung theo kiểu chung chung. Phải trích nguyên văn đoạn văn phù hợp.**

    6. **Trả lời dưới định dạng JSON**, bao gồm các trường sau:
    - `"summary_answer"`: 
        - Trích nguyên văn các đoạn có liên quan, giữ đúng cấu trúc phân mục ban đầu.
        - Nếu đoạn văn được chọn chứa các mục dạng `a), b), c)` hoặc `1., 2., 3.` hoặc các dòng có mã số như `A.2.29`, `A.2.29.1`, **phải lấy đầy đủ từ đầu đến cuối mục đó và các mục con bên dưới**.

        - Với mỗi văn bản và tiêu đề trong phần `"results"`, trình bày câu trả lời theo định dạng:
        ```
        **Văn bản:** Document 1
        Tiêu đề: Title 1
        - Nội dung liên quan đầy đủ theo định dạng gạch đầu dòng, số mục hoặc mã mục...

        **Văn bản:** Document 2
        Tiêu đề: Title 2
        - ...
        ```
    - `"results"`: Danh sách các cặp `[document, title]`, trong đó:
        - `document`: Tên văn bản, lấy từ tiêu đề `# Văn bản`.
        - `title`: Tiêu đề phần phù hợp nhất, lấy từ `## Tiêu đề`, không được chọn bên trong đoạn văn.
    - `"needAgent"`: `true` nếu cần chuyển cho chuyên viên xử lý.
    - `"needSearch"`: `true` nếu không tìm thấy thông tin trong nội dung tham khảo và cần tra cứu thêm.

    7. **Không được suy luận hoặc thêm thông tin ngoài tài liệu đã cho.**
    8. Nếu khách hàng yêu cầu gặp chuyên viên hoặc câu hỏi vượt quá khả năng xử lý, trả lời:
    > "Tôi sẽ gửi yêu cầu của bạn đến chuyên viên chăm sóc khách hàng. Họ sẽ liên hệ với bạn trong thời gian sớm nhất."
    và đặt `"needAgent": true`.
    9. Nếu không tìm thấy thông tin phù hợp trong nội dung tham khảo, bắt buộc đặt `"needSearch": true` và `"results"` phải là danh sách rỗng.

    🔴 10. **Quan trọng**: Nếu một phần như `6.11` hoặc `A.2.29` được chọn thì phải bao gồm đầy đủ tất cả các tiểu mục từ `6.11.1` đến `6.11.x`, hoặc `A.2.29.1` đến `A.2.29.x`. Tuyệt đối không được cắt bớt.

    **Định dạng JSON trả lời mẫu:**
    ```json
    {{
        "summary_answer": "**Văn bản:** Document 1\\nTiêu đề 1: Title 1\\n- Mục a) ...\\n- Mục b) ...\\n- A.2.29.1 ...\\n- A.2.29.2 ...",
        "results": [
            ["Document 1", "Title 1"]
        ],
        "needAgent": false,
        "needSearch": false
    }}

    Nếu không tìm thấy thông tin phù hợp, bắt buộc phải trả về đúng mẫu:
    {{
        "summary_answer": "",
        "results": [],
        "needAgent": false,
        "needSearch": true
    }}

    Câu hỏi:
    {query}

    Nội dung tham khảo:
    {context_block}
    """
        return prompt

    def make_rag_prompt_eng(self, query: str, context_block: str) -> str:
        prompt = f"""
            You are a professional virtual assistant, tasked with answering customer questions based strictly on the provided technical documents or legal texts below.

            Question:
            {query}

            Reference content:
            {context_block}

            Answer requirements:
            1. Carefully analyze the key terms in the question (including object names, actions, technical specifications, specific regulations, etc.) and **directly compare these with the section/chapter titles in the reference content** to select the most relevant documents and titles. Avoid general or unrelated titles.
            2. If there are multiple relevant titles or sections, **list all of them without omitting any important sections.**
            3. **For any section selected, you must include the entire content of that section, including all subpoints and numbered/lettered items (such as "A.2.29", "A.2.29.1", "6.11.4", "a)", "b)", etc.). Do not omit, summarize, paraphrase, or remove any items. Extract content verbatim.**
            4. Respond in JSON format including:
            - "summary_answer": The **verbatim extracted content** from all selected sections, maintaining the original structure, numbering, and formatting. For each document and section, present the answer as:
                ```
                **Document:** Document 1
                Section: Title 1
                - Full relevant content...

                **Document:** Document 2
                Section: Title 2
                - Full relevant content...
                ```
            - "results": A list of [document, title] pairs, where each pair is the document name (from "# Document") and the section title (from "## Title").
            - "needAgent": true if the question should be escalated to a human agent.
            - "needSearch": true if no relevant information is found and further research is needed.
            5. **Absolutely no inference, summarization, or addition of information not present in the reference text. Only base your answer on the provided content.**
            6. If the customer requests to meet an agent or similar, respond: "I will forward your request to the customer service agent. They will contact you as soon as possible." and set "needAgent": true.
            7. If no relevant information is found in the reference content, set "needSearch": true and "results" as an empty list. Do not answer in any other format.
            8. **Do not remove any subsections or subpoints if a section is selected. All subpoints (e.g., "A.2.29.1", "a)", "b)", etc.) must be included in full, with their original numbering and formatting.**

            Please format your answer in JSON like this example:

            {{
            "results": [
                ["Document 1", "Title 1"],
                ["Document 2", "Title 2"]
            ],
            "needAgent": false,
            "needSearch": false
            }}

            Remember: If no relevant information is found, you must return:

            {{
            "results": [],
            "needAgent": false,
            "needSearch": true
            }}
        """
        return prompt

    def get_full_section_content(self, doc_name, section_title, json_folder: str = "new-vn-data-json"):
        norm_title = self.normalize_title(section_title)
        doc_file = doc_name + ".json"
        doc_path = os.path.join(json_folder, doc_file)

        try:
            with open(doc_path, "r", encoding="utf-8") as f:
                doc_data = json.load(f)
        except FileNotFoundError:
            print(f"⚠️ Warning: JSON file not found, skipping: {doc_path}")
            return f"Không tìm thấy file {doc_path}"

        matching_section = self.find_matching_section_or_subsection(
            doc_data, norm_title)

        if matching_section:
            return "\n".join(self.collect_all_titles_and_texts(matching_section))

        return "[0] Không tìm thấy phần nội dung nào phù hợp với tiêu đề và tài liệu đã cho."

    def normalize_title(self, text: str) -> str:
        text = unicodedata.normalize('NFD', text.strip())
        text = ''.join(c for c in text if unicodedata.category(c)
                       != 'Mn')  # remove accents
        text = ''.join(c for c in text if not c.isspace())  # remove whitespace
        return text.lower()

    def find_matching_section_or_subsection(self, doc_data, norm_title):
        for section in doc_data.get("sections", []):
            result = self._find_in_node(section, norm_title)
            if result:
                return result
        return None

    def _find_in_node(self, node, norm_title):
        section_title = node.get("section") or node.get("subsection")
        if section_title:
            norm_section = self.normalize_title(section_title)
            similarity = difflib.SequenceMatcher(
                None, norm_section, norm_title).ratio()
            if similarity >= 0.9:
                return node

        for child in node.get("content", []):
            result = self._find_in_node(child, norm_title)
            if result:
                return result

        return None

    def collect_all_titles_and_texts(self, node):
        texts = []
        title = node.get("section") or node.get("subsection")
        if title:
            texts.append(title)
        if node.get("type") == "text":
            texts.append(node.get("data", ""))
        for child in node.get("content", []):
            texts.extend(self.collect_all_titles_and_texts(child))
        return texts

    def clean_markdown(self, text: str) -> str:
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text.strip()

    def filter_answer(
        self,
        full_sections,
        answer: str,
        json_folder: str = "new-vn-data-json",
        language: str = "vi"
    ) -> Tuple[str, str]:
        print("\nRaw answer:", answer)
        try:
            if isinstance(answer, str):
                answer = answer.strip().removeprefix("```json").removesuffix("```").strip()
                data = json.loads(answer)
            elif isinstance(answer, dict):
                data = answer
        except json.JSONDecodeError as e:
            print("JSON Decode Error:", str(e))
            return "Cần tìm kiếm thêm thông tin trên mạng. Vui lòng thử lại sau.", ""

        results = data.get("results", [])
        need_agent = data.get("needAgent", False)
        need_search = data.get("needSearch", False)
        summary_answer = data.get("summary_answer", "")

        if need_search:
            return "Cần tìm kiếm thêm thông tin trên mạng. Vui lòng thử lại sau.", ""
        if not results or summary_answer == "":
            return "Không tìm thấy tài liệu hoặc tiêu đề liên quan trong câu trả lời.", ""

        doc_sections = {}
        for doc_title_pair in results:
            if not (isinstance(doc_title_pair, (list, tuple)) and len(doc_title_pair) == 2):
                continue
            document, guessed_title = doc_title_pair
            matched_section_title = None
            matched_body = None
            for section_text in full_sections:
                match = re.search(
                    r"# Văn bản: (.*?)\n## Tiêu đề: (.*?)\n\n Đoạn văn: (.*)", section_text, re.DOTALL)
                if not match:
                    continue
                doc_name, real_title, body = match.groups()
                if doc_name.lower().strip() != document.lower().strip():
                    continue
                if guessed_title.strip() in body:
                    matched_section_title = real_title
                    matched_body = body
                    break
            if matched_section_title:
                norm_title = self.normalize_title(matched_section_title)
            doc_file = document.lower() + ".json"
            doc_path = os.path.join(json_folder, doc_file)
            try:
                with open(doc_path, "r", encoding="utf-8") as f:
                    doc_data = json.load(f)
            except FileNotFoundError:
                continue
            matching_section = self.find_matching_section_or_subsection(
                doc_data, norm_title)
            if matching_section:
                doc_sections.setdefault((document, doc_data.get(
                    "filename", document)), []).append(matching_section)

        if not doc_sections:
            return "[1] Không tìm thấy phần nội dung nào phù hợp với tiêu đề và tài liệu đã cho.", ""

        all_blocks = []
        for (document, display_name), sections in doc_sections.items():
            if language == "vi":
                header = "**Văn bản:** " + display_name.upper()
            else:
                header = "**Document:** " + display_name.upper()
            contents = []
            for section in sections:
                section_text = "\n".join(
                    self.collect_all_titles_and_texts(section))
                contents.append(section_text)
            block = header + "\n\n" + "\n\n".join(contents)
            all_blocks.append(block)

        full_text = "\n\n---\n\n".join(all_blocks)
        if need_agent:
            full_text += "\n\n[Thông báo: Yêu cầu được chuyển đến chuyên viên tư vấn.]"
        summary_answer = self.clean_markdown(summary_answer)
        full_text = self.clean_markdown(full_text)
        return summary_answer, full_text

    def get_relevant_sections_for_query(self, query, text_db, n_results=10, json_folder="new-vn-data-json"):
        """
        Tối ưu chọn context: lấy các section liên quan đến chịu lửa ống gió, chịu lửa EI, kể cả khi LLM không match đúng tiêu đề.
        - Kết hợp semantic (vector db) và keyword rule.
        - Ưu tiên lấy cả những đoạn chứa các từ khóa kỹ thuật "giới hạn chịu lửa", "EI", "ống gió", "ống dẫn khí", "kênh dẫn", ...
        """
        # Định nghĩa từ khóa kỹ thuật liên quan chịu lửa
        fire_keywords = [
            "giới hạn chịu lửa", "EI", "chịu lửa", "ống gió", "kênh dẫn", "ống dẫn khí",
            "van ngăn khói", "van ngăn cháy", "phải đảm bảo", "đáp ứng", "REI", "EI"
        ]
        # Lowercase hóa truy vấn để so khớp
        query_lc = query.lower()
        # Lấy n_results đầu từ vector db
        text_res = text_db.query(query_texts=[query], n_results=n_results, include=[
                                 "documents", "metadatas", "distances"])
        text_doc = text_res['documents'][0]
        text_metadata = text_res['metadatas'][0]
        picked_sections = []

        for doc_text, meta in zip(text_doc, text_metadata):
            doc_name = meta.get("doc_name", "Unknown file")
            section_title = meta.get("section", "No Section Title")
            # Tải toàn bộ nội dung section (full text, cả tiêu đề và các tiểu mục)
            section_full_text = self.get_full_section_content(
                doc_name, section_title, json_folder=json_folder)
            section_text_lc = section_full_text.lower()
            section_title_lc = section_title.lower()
            # Nếu section hoặc nội dung có chứa từ khóa kỹ thuật, luôn thêm vào context
            if any(kw in section_title_lc or kw.lower() in section_text_lc for kw in fire_keywords):
                section_str = f"# Văn bản: {doc_name.upper()}\n## Tiêu đề: {section_title}\n\n Đoạn văn: {section_full_text}"
                picked_sections.append(section_str)
            # Nếu similarity tiêu đề với truy vấn lớn hơn 0.7 cũng lấy
            elif difflib.SequenceMatcher(None, section_title_lc, query_lc).ratio() > 0.7:
                section_str = f"# Văn bản: {doc_name.upper()}\n## Tiêu đề: {section_title}\n\n Đoạn văn: {section_full_text}"
                picked_sections.append(section_str)

        # Loại bỏ duplicate (bằng nội dung đoạn)
        uniq_sections = []
        seen_hashes = set()
        for sec in picked_sections:
            sec_hash = hash(sec)
            if sec_hash not in seen_hashes:
                uniq_sections.append(sec)
                seen_hashes.add(sec_hash)
        return uniq_sections

    def combined_answer(self, text_db, bm25_plus, full_documents_bm25, query, text_n_results=7, json_folder="new-vn-data-json", language="vi"):
        t0 = time.time()
        # 1. Normalize ONCE
        if language == "vi":
            normalized_query = self.normalize_viet(query)
        else:
            normalized_query = self.normalize_eng(query)
        t1 = time.time()
        print(f"⏱️ Time to normalize query: {t1 - t0:.2f}s")

        # 2. Retrieve ONCE
        text_res = text_db.query(query_texts=[normalized_query], n_results=text_n_results, include=[
                                 "documents", "metadatas", "distances"])
        t2 = time.time()
        print(f"⏱️ Vector DB query time: {t2 - t1:.2f}s")

        # 3. Assemble context blocks
        # full_sections = []
        # text_doc = text_res['documents'][0]
        # text_metadata = text_res['metadatas'][0]
        # for doc_text, meta in zip(text_doc, text_metadata):
        #     doc_name = meta.get("doc_name", "Unknown file")
        #     section_title = meta.get("section", "No Section Title")
        #     section_text = self.get_full_section_content(
        #         doc_name, section_title)
        #     if language == "vi":
        #         section_text = f"# Văn bản: {doc_name}\n## Tiêu đề: {section_title}\n\n Đoạn văn: {section_text}"
        #     else:
        #         section_text = f"# Document: {doc_name}\n## Title: {section_title}\n\n Text: {section_text}"
        #     if section_text:
        #         full_sections.append(section_text)
        # combined_context = "\n----\n".join(full_sections)

        # 3. Assemble context blocks
        full_sections = self.get_relevant_sections_for_query(
            normalized_query, text_db, n_results=text_n_results, json_folder=json_folder
        )
        combined_context = "\n----\n".join(full_sections)

        # 4. Prompt LLM ONCE
        if language == "vi":
            prompt = self.make_rag_prompt_viet(query, combined_context)
        else:
            prompt = self.make_rag_prompt_eng(query, combined_context)

        answer = self.generate_model_answer(prompt)
        t3 = time.time()
        print(f"⏱️ LLM answer time: {t3 - t2:.2f}s")

        # 5. Filter/format answer (if required)
        summary_answer, references = self.filter_answer(
            full_sections, answer, json_folder=json_folder, language=language)
        t4 = time.time()
        print(f"⏱️ Total time: {t4 - t0:.2f}s")

        return summary_answer, references
