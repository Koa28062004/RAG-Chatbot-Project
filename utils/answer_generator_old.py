import re
import difflib
import time
import unicodedata
import os
import json
import numpy as np
from pyvi import ViTokenizer

from collections import defaultdict
from typing import Dict, Tuple

from config import Config
from utils.search_query import QueryProcessor
from typing import List
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

    # def generate_openai_answer(self, prompt: str) -> str:
    #     response = self.model.chat.completions.create(
    #         model="gpt-4o-mini",
    #         messages=[{"role": "user", "content": prompt}],
    #         max_tokens=1000
    #     )
    #     return response.choices[0].message.content.strip()

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

        Ví dụ:
        Truy vấn gốc: "Hành lang bên là gì"  
        Truy vấn chuẩn hóa: "Theo quy chuẩn, tiêu chuẩn hiện hành về phòng cháy chữa cháy, hành lang bên của nhà hoặc công trình được định nghĩa như thế nào và có yêu cầu gì về kích thước, vật liệu xây dựng, và khả năng chịu lửa?"

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
    4. Nếu câu trả lời có table, hãy format lại đúng định dạng Markdown của bảng, bao gồm các tiêu đề cột và hàng. Đảm bảo giữ nguyên cấu trúc bảng.

    🔴 4. **Bắt buộc giữ nguyên toàn bộ các nội dung có cấu trúc phân mục như `a)`, `b)`, `1.`, `2.`, hoặc mã số như `A.2.29.1`, `6.11.4`... Nếu một mục như `A.2.29` được chọn, phải bao gồm toàn bộ các tiểu mục `A.2.29.x` bên dưới. Không được bỏ sót hoặc rút gọn.**

    🔴 5. **Không được rút gọn, tóm tắt, gom nhóm hoặc diễn giải lại nội dung theo kiểu chung chung. Phải trích nguyên văn đoạn văn phù hợp.**

    🔴 6. Nếu đoạn văn có chứa hình ảnh Markdown (ví dụ: ![caption](path/to/image.png)), bắt buộc phải giữ nguyên định dạng hình ảnh này trong 'summary_answer'. Không được xóa hoặc thay đổi đường dẫn hình ảnh.
    
    🔴 7. Nếu câu hỏi có tính chất tổng quát như "phân loại nhóm nhà", phải mở rộng tìm kiếm đến tất cả các tiêu đề và nội dung liên quan đến khái niệm chính (ví dụ: công năng, chiều cao, kết cấu, vật liệu...). Không chỉ chọn phần chứa đúng cụm từ câu hỏi, mà phải bao quát toàn bộ các tiêu chí phân loại liên quan. Phải giữ url ảnh markdown.

    8. **Trả lời dưới định dạng JSON**, bao gồm các trường sau:
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

    9. **Không được suy luận hoặc thêm thông tin ngoài tài liệu đã cho.**
    10. Nếu khách hàng yêu cầu gặp chuyên viên hoặc câu hỏi vượt quá khả năng xử lý, trả lời:
    > "Tôi sẽ gửi yêu cầu của bạn đến chuyên viên chăm sóc khách hàng. Họ sẽ liên hệ với bạn trong thời gian sớm nhất."
    và đặt `"needAgent": true`.
    11. Nếu không tìm thấy thông tin phù hợp trong nội dung tham khảo, bắt buộc đặt `"needSearch": true` và `"results"` phải là danh sách rỗng.

    🔴 12. **Quan trọng**: Nếu một phần như `6.11` hoặc `A.2.29` được chọn thì phải bao gồm đầy đủ tất cả các tiểu mục từ `6.11.1` đến `6.11.x`, hoặc `A.2.29.1` đến `A.2.29.x`. Tuyệt đối không được cắt bớt.
    
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

    def voted_original_normalized_answer(self, query, summary_answer, normalized_answer):
        prompt = f"""
    Bạn là một chuyên gia trong lĩnh vực Phòng cháy chữa cháy (PCCC), am hiểu sâu sắc các tài liệu pháp lý và kỹ thuật như Quy chuẩn (QC), Tiêu chuẩn (TCVN), Nghị định, Thông tư,...

    Dưới đây là một câu hỏi gốc của người dùng và hai câu trả lời được tạo ra từ hai phiên bản khác nhau của câu hỏi: bản gốc và bản đã được chuẩn hoá lại để tìm kiếm tốt hơn.

    ## Câu hỏi gốc:
    {query}

    ## Trả lời từ câu hỏi gốc (dưới dạng JSON):
    ```json
    {summary_answer}

    ## Trả lời từ câu hỏi đã chuẩn hoá (dưới dạng JSON):
    ```json
    {normalized_answer}

    ## Nhiệm vụ của bạn:

    1. Đọc kỹ câu hỏi gốc để xác định nội dung chính cần trả lời.
    2. Đánh giá nội dung của cả hai câu trả lời và **tổng hợp lại các phần phù hợp nhất với câu hỏi gốc**, đảm bảo **không bỏ sót hoặc tóm tắt bất kỳ mục, phân mục hoặc tiểu mục nào**. Không được bỏ qua các đánh số, ký hiệu mục như "A.2.29", "A.2.29.1", "6.11.4", "a)", "b)",... Nếu một mục hoặc phân mục được chọn, bạn phải **giữ nguyên toàn bộ nội dung và các mục con bên dưới**, không được rút gọn hoặc bỏ qua.
    3. Không chỉ sao chép, cũng không được cắt xén, mà phải **giữ lại toàn bộ các nội dung liên quan, bảo toàn cấu trúc gốc, bao gồm mọi phân mục và đánh số/formatting ban đầu**.
    4. Trả về kết quả dưới dạng JSON với các trường sau:
        - "summary_answer": câu trả lời mới được tổng hợp lại, **trích nguyên văn, giữ đúng định dạng markdown và mọi phân mục, tiểu mục** như ban đầu.
        - "results": danh sách các cặp [document_id, document_title] liên quan trực tiếp đến phần nội dung được giữ lại.
        - "needAgent" và "needSearch": giữ nguyên hoặc cập nhật logic nếu một trong hai câu trả lời có true.

    ## Định dạng JSON mẫu:
    ```json
    {{
        "summary_answer": "**Văn bản:** Document X\\n- Đầy đủ nội dung, giữ nguyên mọi phân mục, tiểu mục...\\n- A.2.29 ...\\n- A.2.29.1 ...",
        "results": [
            ["Document X", "Tiêu đề X"],
            ["Document Y", "Tiêu đề Y"]
        ],
        "needAgent": false,
        "needSearch": false
    }}
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

    def make_rag_prompt_with_history(query: str, context_block: List[str], history: List[str]) -> str:
        history_text = "\n".join(history[:-5]) if history else ""
        context_block = "\n---\n".join(context_block)

        prompt = f"""
        Bạn là một trợ lý ảo chăm sóc khách hàng.

        Lịch sử trò chuyện:
        {history_text}

        Nội dung tham khảo:
        {context_block}

        Câu hỏi:
        {query}

        Hãy trả lời một cách lịch sự, chu đáo và chính xác:
        - Nếu khách hàng nói "tôi muốn gặp chuyên viên" hoặc tương tự, trả lời: "Tôi sẽ gửi yêu cầu của bạn đến chuyên viên chăm sóc khách hàng. Họ sẽ liên hệ với bạn trong thời gian sớm nhất."
        - Nếu có thông tin phù hợp, trả lời đầy đủ và chính xác, không thêm suy diễn ngoài nội dung tham khảo.
        - Nếu không tìm thấy thông tin, đề nghị khách hàng liên hệ nhân viên tư vấn.

        Vui lòng định dạng câu trả lời dưới dạng JSON:
        {{
            "response": "<Câu trả lời>",
            "needAgent": <true/false>,
        }}
    """
        return prompt

    def get_full_section_content(self, doc_name, section_title, json_folder: str = "new-vn-data-json"):
        # ✅ 1) Group matches by document
        doc_sections = {}  # dict: { document : [sections...] }

        doc_file = doc_name + ".json"
        doc_path = os.path.join(json_folder, doc_file)

        try:
            with open(doc_path, "r", encoding="utf-8") as f:
                doc_data = json.load(f)
        except FileNotFoundError:
            print(f"⚠️ Warning: JSON file not found, skipping: {doc_path}")
            return f"Không tìm thấy file {doc_path}"

        norm_title = self.normalize_title(section_title)
        matching_section = self.find_matching_section_or_subsection(
            doc_data, norm_title)

        if matching_section:
            doc_sections.setdefault((doc_name, doc_data.get(
                "filename", doc_name)), []).append(matching_section)
        else:
            print(f"❌ Không tìm thấy tiêu đề phù hợp cho: '{section_title}' (chuẩn hoá: '{norm_title}')")

        if not doc_sections:
            return "Không tìm thấy phần nội dung nào phù hợp với tiêu đề và tài liệu đã cho."

        # ✅ 2) Build blocks, one per document, all its sections inside
        all_blocks = []
        for (document, display_name), sections in doc_sections.items():
            # Gather texts of all matched sections
            contents = []
            for section in sections:
                section_text = "\n".join(
                    self.collect_all_titles_and_texts(section))
                contents.append(section_text)

            block = "\n\n".join(contents)
            all_blocks.append(block)

        # ✅ 3) Join blocks with ---
        full_text = "\n\n---\n\n".join(all_blocks)

        return full_text

    def strip_accents(text):
        return ''.join(
            c for c in unicodedata.normalize('NFD', text)
            if unicodedata.category(c) != 'Mn'
        )

    def normalize_title(self, text: str) -> str:
        text = unicodedata.normalize('NFD', text.strip())
        text = ''.join(c for c in text if unicodedata.category(c)
                       != 'Mn')  # remove accents
        # remove all whitespace chars
        text = ''.join(c for c in text if not c.isspace())
        return text.lower()

    def find_matching_section_or_subsection(self, doc_data, norm_title):
        """
        Search through all top-level sections and their nested content 
        to find a section or subsection whose normalized title matches norm_title.
        """
        for section in doc_data.get("sections", []):
            result = self._find_in_node(section, norm_title)
            if result:
                return result
        return None

    def _find_in_node(self, node, norm_title):
        """
        Recursively check this node and its children for a similarity match >= 80%.
        """
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

    def filter_answer(
        self,
        full_sections,
        answer: str,
        json_folder: str = "new-vn-data-json",
        language: str = "vi"
    ) -> Tuple[str, str]:

        # answer = answer.strip().removeprefix("```json").removesuffix("```").strip()

        print("\nRaw answer:", answer)

        try:
            if isinstance(answer, str):
                answer = answer.strip().removeprefix("```json").removesuffix("```").strip()
                data = json.loads(answer)
            elif isinstance(answer, dict):
                data = answer
        except json.JSONDecodeError as e:
            print("JSON Decode Error:", str(e))
            return "Cần tìm kiếm thêm thông tin trên mạng. Vui lòng thử lại sau."

        results = data.get("results", [])
        need_agent = data.get("needAgent", False)
        need_search = data.get("needSearch", False)
        summary_answer = data.get("summary_answer", "")

        print("\nSUMMARY ANSWER: ", summary_answer)

        if need_search:
            return "Cần tìm kiếm thêm thông tin trên mạng. Vui lòng thử lại sau.", ""
        if not results or summary_answer == "":
            return "Không tìm thấy tài liệu hoặc tiêu đề liên quan trong câu trả lời.", ""

        # ✅ 1) Group matches by document
        doc_sections = {}  # dict: { (document, filename) : [sections...] }

        for doc_title_pair in results:
            if not (isinstance(doc_title_pair, (list, tuple)) and len(doc_title_pair) == 2):
                continue

            document, guessed_title = doc_title_pair
            matched_section_title = None
            matched_body = None

            # ✅ Dò lại đúng tiêu đề từ full_sections
            for section_text in full_sections:
                match = re.search(
                    r"# Văn bản: (.*?)\n## Tiêu đề: (.*?)\n\nĐoạn văn: (.*)", section_text, re.DOTALL)
                if not match:
                    continue
                doc_name, real_title, body = match.groups()

                print(f"Checking document: {doc_name}, guessed title: {guessed_title}, real title: {real_title}")

                # Match đúng văn bản
                if doc_name.strip() != document.strip():
                    continue

                if guessed_title.strip() == real_title.strip():
                    matched_section_title = real_title
                    matched_body = body
                    break

                # ✅ Fallback: Nếu guessed_title nằm trong nội dung đoạn văn
                if guessed_title.strip() in body:
                    matched_section_title = real_title
                    matched_body = body
                    break

            if matched_section_title:
                print(
                    f"Matched section title: {matched_section_title} for document: {document}")
                norm_title = self.normalize_title(matched_section_title)
            else:
                print(f"[⚠️] Không tìm được matched_section_title cho document: {document}, guessed_title: {guessed_title}")
                continue

            doc_file = document + ".json"
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
            return "Không tìm thấy phần nội dung nào phù hợp với tiêu đề và tài liệu đã cho.", ""

        # ✅ 2) Build blocks, one per document, all its sections inside
        all_blocks = []
        for (document, display_name), sections in doc_sections.items():
            if language == "vi":
                header = "**Văn bản:** " + display_name
            else:
                header = "**Document:** " + display_name

            contents = []
            for section in sections:
                section_text = "\n".join(
                    self.collect_all_titles_and_texts_section(section))
                contents.append(section_text)

            block = header + "\n\n" + "\n\n".join(contents)
            all_blocks.append(block)

        # ✅ 3) Join blocks with ---
        full_text = "\n\n---\n\n".join(all_blocks)

        if need_agent:
            full_text += "\n\n[Thông báo: Yêu cầu được chuyển đến chuyên viên tư vấn.]"

        # ✅ 4) Clean markdown
        summary_answer = self.clean_markdown(summary_answer)
        full_text = self.clean_markdown(full_text)

        # ✅ 5) Đổi tên **Văn bản:** trong summary_answer thành filename thực tế
        docname_to_filename = {}
        for (document, display_name), _ in doc_sections.items():
            docname_to_filename[document.strip()] = display_name
            docname_to_filename[display_name.strip()] = display_name

        def replace_doc_names(match):
            old_docname = match.group(1).strip()
            new_docname = docname_to_filename.get(old_docname, old_docname)
            print("Replacing docname:", old_docname, "→", new_docname)
            return f"**Văn bản:** {new_docname}"

        summary_answer = re.sub(
            r"\*\*Văn bản:\*\*\s*(.*?)\s*(?=\n|$)", replace_doc_names, summary_answer)

        # ✅ 6) Ưu tiên sắp xếp các block theo QC → TC → khác
        blocks = re.split(r'(?=\*\*Văn bản:\*\*)', summary_answer)

        def get_priority(block):
            match = re.search(r"\*\*Văn bản:\*\*\s*(.*?)\s*(?=\n|$)", block)
            if not match:
                return 3  # nếu không rõ thì ưu tiên thấp nhất
            name = match.group(1).strip().upper()
            if name.startswith("QCVN06"):
                return 0
            elif name.startswith("QC"):
                return 1
            elif name.startswith("TC"):
                return 2
            else:
                return 3

        blocks_sorted = sorted(blocks, key=get_priority)
        summary_answer = "\n".join(blocks_sorted).strip()

        return summary_answer, full_text

    def clean_markdown(self, text: str) -> str:
        # Replace 3+ newlines with 2
        text = re.sub(r'\n{3,}', '\n\n', text)
        # Remove leading/trailing newlines
        return text.strip()

    def collect_all_titles_and_texts_section(self, node):
        """
        Recursively collect:
        - the node's own section/subsection title (if any)
        - all text data inside
        - all nested sections/subsections titles too
        """
        texts = []

        # # ✅ 1) Include the current section/subsection title as text with leading \n
        title = node.get("section") or node.get("subsection")
        if title:
            texts.append("\n" + title)

        # ✅ 2) Include the current text node, if any, with \n if no '-' or '+' at start
        if node.get("type") == "text":
            data = node.get("data", "")
            if "|" in data:
                texts.append(data)
            elif data.lstrip().startswith(("-", "+")):
                texts.append(data)
            else:
                texts.append("\n" + data)
        elif node.get("type") == "image":
            image_md = node.get("image_markdown", "")
            image_desc = node.get("image_name", "Image")
            texts.append(f"![{image_desc}]({image_md})")

        # ✅ 3) Recurse into children
        for child in node.get("content", []):
            texts.extend(self.collect_all_titles_and_texts_section(child))

        return texts

    def collect_all_titles_and_texts(self, node, level=0):
        """
        Recursively collect:
        - Skip the top-level section/subsection title (level=0)
        - Include all nested section/subsection titles (level >= 1)
        - Collect all text and image nodes
        """
        texts = []

        # ✅ 1) Include title only if it's not the top-level one
        title = node.get("section") or node.get("subsection")
        if title and level > 0:
            texts.append("\n" + title)

        # ✅ 2) Include the current text node, if any
        if node.get("type") == "text":
            data = node.get("data", "")
            if "|" in data:
                texts.append(data)
            elif data.lstrip().startswith(("-", "+")):
                texts.append(data)
            else:
                texts.append("\n" + data)
        elif node.get("type") == "image":
            image_md = node.get("image_markdown", "")
            image_desc = node.get("image_name", "Image")
            texts.append(f"![{image_desc}]({image_md})")

        # ✅ 3) Recurse into children with increased level
        for child in node.get("content", []):
            texts.extend(self.collect_all_titles_and_texts(child, level=level+1))

        return texts

    def collect_all_content_with_headings(self, section, level=1):
        """
        Recursively collect all content with Markdown headings for sections/subsections.
        - Starts with level=1 (which means "#").
        - Each deeper subsection adds one more "#".
        """

        lines = []

        # Determine heading key: section or subsection
        heading = section.get("section") or section.get(
            "subsection") or "No Title"

        # Add heading with appropriate #
        heading_prefix = "#" * level
        lines.append(f"\n{heading}")

        # Add content (text or images)
        for item in section.get("content", []):
            # If this item is a nested subsection -> recurse
            if "subsection" in item:
                lines.extend(self.collect_all_content_with_headings(
                    item, level=level + 1))
            else:
                if item["type"] == "text":
                    lines.append(item["data"])
                elif item["type"] == "image":
                    image_md = item.get("image_markdown", "")
                    image_desc = item.get("image_name", "Image")
                    lines.append(f"![{image_desc}]({image_md})")

        return lines

    def search_bm25(self, bm25_plus, full_documents_bm25, query, text_n_results=2):
        bm25_embedding_fn = BM25EmbeddingFunction()
        query_tokens = bm25_embedding_fn.bm25_tokenizer(query)
        scores = bm25_plus.get_scores(query_tokens)

        top_indices = np.argsort(scores)[::-1][:text_n_results]
        top_results = [full_documents_bm25[i] for i in top_indices]

        # print("\n\nBM25 search results:\n")
        # for i, obj in enumerate(top_results):
        #     meta = obj.get("metadata", {})
        #     doc_name = meta.get("doc_name", "Unknown file")
        #     section_title = meta.get("section", "No Section Title")
        #     print(f"\n[BM25 chunk {i}] from file: {doc_name} - Title: {section_title} - Score: {scores[top_indices[i]]:.4f}")

        return top_results

    def parse_summary_answer_blocks(self, summary_text: str) -> List[Tuple[str, str, str]]:
        blocks = []
        current_doc_id = None
        current_title = None
        current_content_lines = []

        lines = summary_text.strip().splitlines()

        for line in lines:
            if line.startswith("**Văn bản:**"):
                # Save previous block
                if current_doc_id is not None:
                    content = "\n".join(current_content_lines).strip()
                    blocks.append((current_doc_id, current_title, content))
                # Start new block
                current_doc_id = line.replace("**Văn bản:**", "").strip()
                current_title = None
                current_content_lines = []
            elif "Tiêu đề:" in line:
                raw_title = line.split("Tiêu đề:", 1)[-1].strip()
                # Remove leading '**', '*', or extra spaces
                cleaned_title = re.sub(r"^\*+\s*", "", raw_title).strip()
                current_title = cleaned_title
                current_content_lines.append(line.strip())  # keep original for context
            else:
                current_content_lines.append(line.strip())

        # Save the last block
        if current_doc_id is not None:
            content = "\n".join(current_content_lines).strip()
            blocks.append((current_doc_id, current_title, content))

        return blocks
    
    def merge_answers(
        self,
        original_answer: dict,
        normalized_answer: dict,
        original_full_sections: List[str],
        normalized_full_sections: List[str]
    ) -> Tuple[dict, List[str]]:

        def parse_json_block(block):
            try:
                cleaned = block.strip().removeprefix("```json").removesuffix("```").strip()
                return json.loads(cleaned)
            except json.JSONDecodeError as e:
                print("❌ JSON decode error:", e)
                print("🔎 Raw content that caused error:\n", block[:500])
                return {}

        # Step 1: Parse JSON
        original_json = parse_json_block(original_answer) if original_answer else {}
        normalized_json = parse_json_block(normalized_answer) if normalized_answer else {}

        # Step 2: Combine and deduplicate results
        seen = set()
        combined_results = []
        for result in original_json.get("results", []) + normalized_json.get("results", []):
            key = tuple(result)
            if key not in seen:
                seen.add(key)
                combined_results.append(result)

        # Step 3: Get (doc_id, title) from results
        relevant_keys = {
            tuple(result[:2]) for result in combined_results
            if isinstance(result, list) and len(result) >= 2
        }

        # Step 4: Parse summary blocks
        original_blocks = self.parse_summary_answer_blocks(original_json.get("summary_answer", ""))
        normalized_blocks = self.parse_summary_answer_blocks(normalized_json.get("summary_answer", ""))

        for doc_id, title, content in original_blocks + normalized_blocks:
            print(f"✅ Parsed: {doc_id=} | {title=}")

        # Step 5: Merge summary blocks based on (doc_id, title)
        summary_blocks = []
        added_keys = set()

        ordered_results = original_json.get("results", []) + normalized_json.get("results", [])
        for result in ordered_results:
            if not isinstance(result, list) or len(result) < 2:
                continue
            doc_id, title = result[:2]
            key = (doc_id, title)
            if key in added_keys:
                continue
            added_keys.add(key)

            found = False
            for blocks in [original_blocks, normalized_blocks]:
                for block_doc_id, block_title, content in blocks:
                    if (block_doc_id, block_title) == key:
                        full_block = f"**Văn bản:** {block_doc_id}\n{content}"
                        summary_blocks.append(full_block)
                        found = True
                        break
                if found:
                    break

        # Deduplicate summary blocks
        seen_summary_hashes = set()
        dedup_summary_blocks = []
        for blk in summary_blocks:
            blk_hash = hash(blk)
            if blk_hash not in seen_summary_hashes:
                seen_summary_hashes.add(blk_hash)
                dedup_summary_blocks.append(blk)

        summary_combined = "\n\n".join(dedup_summary_blocks).strip()

        # Step 6: Merge full_sections based on doc_id only
        all_sections = original_full_sections + normalized_full_sections
        merged_full_sections = []
        added_section_hashes = set()

        relevant_doc_ids = {doc_id for doc_id, _ in relevant_keys}

        for section in all_sections:
            lines = section.strip().splitlines()
            doc_id = None
            for line in lines:
                if line.lower().startswith("# văn bản:"):
                    doc_id = line.split(":", 1)[-1].strip()
                    break
            if doc_id and doc_id in relevant_doc_ids:
                section_hash = hash(section)
                if section_hash not in added_section_hashes:
                    added_section_hashes.add(section_hash)
                    merged_full_sections.append(section)

        # Step 7: Merge flags
        need_agent = original_json.get("needAgent", False) and normalized_json.get("needAgent", False)
        need_search = original_json.get("needSearch", False) and normalized_json.get("needSearch", False)

        merged_answer = {
            "summary_answer": summary_combined,
            "results": combined_results,
            "needAgent": need_agent,
            "needSearch": need_search,
        }

        return merged_answer, merged_full_sections

    def combined_answer(self, text_db, bm25_plus, full_documents_bm25, query, text_n_results=7, json_folder: str = "new-vn-data-json", language: str = "vi"):
        t0 = time.time()

        if language == "vi":
            normalized_query = self.normalize_viet(query)
        else:
            normalized_query = self.normalize_eng(query)

        print("ORIGINAL QUERY")
        t1 = time.time()
        # summary_answer, references = self.generate_answer(text_db, bm25_plus, full_documents_bm25, query)
        original_answer, original_full_sections = self.generate_answer(
            text_db, bm25_plus, full_documents_bm25, query, text_n_results=4)
        t2 = time.time()
        print(f"⏱️ Time to generate original_answer: {t2 - t1:.2f}s")

        print("NORMALIZED QUERY")
        # normalized_answer, normalized_references = self.generate_answer(text_db, bm25_plus, full_documents_bm25, normalized_query)
        t3 = time.time()
        normalized_answer, normalized_full_sections = self.generate_answer(
            text_db, bm25_plus, full_documents_bm25, normalized_query, text_n_results=4)
        t4 = time.time()
        print(f"⏱️ Time to generate normalized_answer: {t4 - t3:.2f}s")

        # prompt = self.voted_original_normalized_answer(query, original_answer, normalized_answer)
        # voted_answer = self.generate_model_answer(prompt)

        t5 = time.time()
        voted_answer, voted_full_sections = self.merge_answers(
            original_answer, normalized_answer, original_full_sections, normalized_full_sections)
        t6 = time.time()
        print(f"⏱️ Time to merge answers: {t6 - t5:.2f}s")

        print("\nVOTED_ANSWER", voted_answer)

        t7 = time.time()
        summary_answer, references = self.filter_answer(
            voted_full_sections, voted_answer, json_folder=json_folder, language=language)
        t8 = time.time()
        print(f"⏱️ Time to filter answer: {t8 - t7:.2f}s")

        total_time = time.time() - t0
        print(f"✅ Total time for combined_answer: {total_time:.2f}s")

        return summary_answer, references

    def generate_answer(self, text_db, bm25_plus, full_documents_bm25, query, text_n_results=4, json_folder: str = "new-vn-data-json", language: str = "vi"):
        t0 = time.time()

        # segmented_question = query
        segmented_question = ViTokenizer.tokenize(query)
        text_res = text_db.query(query_texts=[segmented_question], n_results=text_n_results, include=[
                                 "documents", "metadatas", "distances"])
        t1 = time.time()
        print(f"⏱️ Vector DB query time: {t1 - t0:.2f}s")

        text_doc = text_res['documents'][0]
        text_metadata = text_res['metadatas'][0]
        text_distances = text_res['distances'][0]

        # Search for relevant sections in the BM25 database
        t2 = time.time()
        bm25_results = self.search_bm25(
            bm25_plus, full_documents_bm25, segmented_question, text_n_results=text_n_results)
        t3 = time.time()
        print(f"⏱️ BM25 search time: {t3 - t2:.2f}s")

        section_keys = set()
        full_sections = []

        # Trace back through the vector database results
        for i, (doc_text, meta, text_dis) in enumerate(zip(text_doc, text_metadata, text_distances)):
            doc_name = meta.get("doc_name", "Unknown file")
            section_title = meta.get("section", "No Section Title")
            key = (doc_name, section_title)

            if key not in section_keys:
                section_keys.add(key)
                # print(f"\n[Text chunk {i}] from file: {doc_name} - Title: {section_title} - Distance: {text_dis:.4f}")
                section_text = self.get_full_section_content(
                    doc_name, section_title)
                if language == "vi":
                    section_text = f"# Văn bản: {doc_name}\n## Tiêu đề: {section_title}\n\nĐoạn văn: {section_text}"
                else:
                    section_text = f"# Document: {doc_name}\n## Title: {section_title}\n\nText: {section_text}"

                if section_text:
                    full_sections.append(section_text)

        # Trace back through the BM25 results
        for i, obj in enumerate(bm25_results, text_n_results):
            meta = obj.get("metadata", {})
            doc_name = meta.get("doc_name", "Unknown file")
            section_title = meta.get("section", "No Section Title")
            key = (doc_name, section_title)

            if key not in section_keys:
                section_keys.add(key)
                bm25_text = obj.get("text", "")
                section_text = self.get_full_section_content(doc_name, section_title)
                if language == "vi":
                    section_text = f"# Văn bản: {doc_name}\n## Tiêu đề: {section_title}\n\nĐoạn văn:\n{section_text}"
                else:
                    section_text = f"# Document: {doc_name}\n## Title: {section_title}\n\nText: {section_text}"

                if section_text:
                    full_sections.append(section_text)

        # print(f"KONIS -- generate_answer -- full_sections {full_sections}")

        combined_context = "\n----\n".join(full_sections)

        if language == "vi":
            prompt = self.make_rag_prompt_viet(query, combined_context)
        else:
            prompt = self.make_rag_prompt_eng(query, combined_context)

        # Truncate the prompt to fit within the model's token limit
        if self.config.MODEL_USED == "openai":
            prompt = prompt[:200000]
        elif self.config.MODEL_USED == "gemini":
            prompt = prompt

        print(f"\n\nPrompt:\n{prompt}\n\n")

        t4 = time.time()
        answer = self.generate_model_answer(prompt)
        t5 = time.time()
        print(f"⏱️ Time to generate model answer: {t5 - t4:.2f}s")

        print(f"Query: {query}")
        print(f"Answer: {answer}")

        total_time = time.time() - t0
        print(f"✅ Total time for generate_answer: {total_time:.2f}s")

        return answer, full_sections

    def generate_answer_with_source(self, text_db, image_db, table_db, query, text_n_results=25, image_n_results=3, table_n_results=3):

        query = self.normalize(query)

        text_res = text_db.query(query_texts=[query], n_results=text_n_results, include=[
                                 "documents", "metadatas", "distances"])
        image_res = image_db.query(query_texts=[query], n_results=image_n_results, include=[
                                   "documents", "metadatas", "distances"])
        table_res = table_db.query(query_texts=[query], n_results=table_n_results, include=[
                                   "documents", "metadatas", "distances"])
        text_doc = text_res['documents'][0]
        image_doc = image_res['documents'][0]
        table_doc = table_res['documents'][0]

        text_metadata = text_res['metadatas'][0]
        image_metadata = image_res['metadatas'][0]
        table_metadata = table_res['metadatas'][0]

        text_distances = text_res['distances'][0]
        image_distances = image_res['distances'][0]
        table_distances = table_res['distances'][0]

        prompt = self.make_rag_prompt(query, text_doc)
        answer = self.generate_model_answer(prompt)

        normalized = unicodedata.normalize("NFKC", answer or "").lower()

        if any(trigger in normalized for trigger in self.FALLBACK_TRIGGERS):
            print("🔁 Trigger fallback: No answer found in response.")
            answer = self.query_processor.search_and_answer(query)

        images_res = []
        tables_res = []

        for i, (doc_text, meta, text_dis) in enumerate(zip(text_doc, text_metadata, text_distances)):
            source_file = meta.get("filename", "Unknown file")
            print(
                f"\n[Text chunk {i}] from file: {source_file} - Distance: {text_dis}\nText:\n{doc_text}\n{'-'*40}")

        for i, (doc_text, meta, image_dis) in enumerate(zip(image_doc, image_metadata, image_distances)):
            source_file = meta.get("url", "Unknown file")
            if image_dis < 0.1:
                images_res.append(source_file)
            print(
                f"\n[Image chunk {i}] from file: {source_file} - Distance: {image_dis}\nText:\n{doc_text}\n{'-'*40}")

        for i, (doc_text, meta, table_dis) in enumerate(zip(table_doc, table_metadata, table_distances)):
            source_file = meta.get("url", "Unknown file")
            if table_dis < 0.1:
                tables_res.append(source_file)
            print(
                f"\n[Table chunk {i}] from file: {source_file} - Distance: {table_dis}\nText:\n{doc_text}\n{'-'*40}")

        answer += "\n\n"
        if images_res and tables_res:
            answer = "Tham khảo hình ảnh và bảng sau:\n"
            # for img in images_res:
            #     answer += f"- Hình ảnh: {img}\n"
            # for table in tables_res:
            #     answer += f"- Bảng: {table}\n"
        elif images_res:
            answer = "Tham khảo hình ảnh sau:\n"
            # for img in images_res:
            #     answer += f"- Hình ảnh: {img}\n"
        elif tables_res:
            answer = "Tham khảo bảng sau:\n"
            # for table in tables_res:
            #     answer += f"- Bảng: {table}\n"

        return (
            query,
            answer,
            images_res,
            tables_res,
            # Convert zip to list
            list(zip(text_doc, text_metadata, text_distances)),
            # Convert zip to list
            list(zip(image_doc, image_metadata, image_distances)),
            # Convert zip to list
            list(zip(table_doc, table_metadata, table_distances))
        )
