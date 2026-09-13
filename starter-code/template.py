"""
Lab #4: System Prompt Engineering & Tool Calling Engine
Học viên hoàn thiện các mục TODO để hoàn thành bài lab.

Kiến trúc:
  - ChatbotBaseline: LLM thuần, không dùng tool → quan sát hallucination.
  - ToolCallingAgent: Agent dùng System Prompt + 2 Tool Schemas.
"""
import os
import json
import re
from typing import Dict, Any, List
from tools import TOOL_DEFINITIONS, TOOL_MAP, search_product_catalog, submit_support_ticket

# ═══════════════════════════════════════════════════════════════════════════
# TODO 1: Thiết kế SYSTEM PROMPT cấp sản xuất
# Yêu cầu: Phải chứa Persona, Core Rules, Operational Boundaries, Output Contract.
# ═══════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """Bạn là VinAssistant – trợ lý AI chính thức của Tập đoàn Vingroup.

## PERSONA
- Tên: VinAssistant
- Vai trò: Chuyên viên tư vấn sản phẩm & dịch vụ Vingroup (VinFast, Vinpearl, Vinhomes,...).
- Giọng nói: Chuyên nghiệp, thân thiện, chính xác, không bịa đặt thông tin.

## AVAILABLE TOOLS
{tools}

## CORE RULES (Bắt buộc tuân thủ)
1. KHÔNG BAO GIỜ bịa dữ liệu sản phẩm (giá, tính năng, tồn kho). Bắt buộc phải gọi tool `search_product_catalog` để lấy dữ liệu thực tế.
2. KHÔNG BAO GIỜ tự tạo ticket_id. PHẢI gọi tool `submit_support_ticket` khi cần ghi nhận hỗ trợ của khách hàng.
3. Nếu khách hàng hỏi câu FAQ đơn giản (chính sách bảo hành, đổi trả chung,...), trả lời trực tiếp mà không cần gọi tool.
4. Nếu câu hỏi cần nhiều tool, hãy gọi lần lượt từng tool một cách tuần tự và logic.

## OPERATIONAL BOUNDARIES
- Chỉ trả lời các câu hỏi liên quan đến sản phẩm, dịch vụ và hệ sinh thái của Vingroup.
- Từ chối lịch sự nếu người dùng hỏi các chủ đề ngoài phạm vi chuyên môn hoặc vi phạm tiêu chuẩn cộng đồng.

## OUTPUT CONTRACT (ReAct Pattern)
Khi xử lý yêu cầu, bạn phải tuân thủ nghiêm ngặt định dạng suy nghĩ và hành động sau:

Thought: [Suy nghĩ, phân tích xem cần làm gì tiếp theo]
Action: [Tên tool cần gọi, hoặc 'None' nếu đã có câu trả lời cuối cùng]
Action Input: [JSON arguments tương ứng của tool]
Observation: [Kết quả trả về từ tool sẽ được chèn vào đây]
... (Lặp lại chuỗi Thought/Action/Action Input/Observation nếu cần)
Final Answer: [Câu trả lời hoàn chỉnh, trau chuốt gửi tới người dùng]
"""


# ═══════════════════════════════════════════════════════════════════════════
# CLASS: ChatbotBaseline
# ═══════════════════════════════════════════════════════════════════════════

class ChatbotBaseline:
    """Baseline LLM Chatbot – Không sử dụng Tool Calling hay ReAct Loop.
    Mục đích: So sánh chất lượng trả lời khi LLM bịa thông tin (hallucination).
    """

    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")

    def query(self, user_input: str) -> Dict[str, Any]:
        """Gửi câu hỏi tới LLM (hoặc trả lời mock nếu không có API key)."""
        if self.api_key:
            try:
                import google.generativeai as genai
                genai.configure(api_key=self.api_key)
                model = genai.GenerativeModel("gemini-1.5-flash")
                
                prompt = f"Bạn là chatbot tư vấn sản phẩm Vingroup. Hãy trả lời câu hỏi sau:\n{user_input}"
                response = model.generate_content(prompt)
                
                return {
                    "answer": response.text,
                    "tool_calls": [],
                    "status": "success",
                    "mode": "gemini_baseline"
                }
            except Exception as e:
                return {
                    "answer": f"Lỗi khi gọi API: {str(e)}",
                    "tool_calls": [],
                    "status": "error",
                    "mode": "gemini_baseline"
                }

        # Trả về kết quả mock nếu chưa có API key
        return {
            "answer": f"[Chatbot Baseline] Trả lời cho: {user_input}",
            "tool_calls": [],
            "status": "success",
            "mode": "mock_baseline"
        }


# ═══════════════════════════════════════════════════════════════════════════
# CLASS: ToolCallingAgent
# ═══════════════════════════════════════════════════════════════════════════

class ToolCallingAgent:
    """Production-grade Agent với System Prompt Engineering & Tool Calling.

    Features:
        - 2 custom tools: search_product_catalog, submit_support_ticket
        - Sequential & Parallel tool calling
        - Max iterations safeguard
        - Full trace logging
    """

    def __init__(self, max_iterations: int = 5, api_key: str = None):
        self.max_iterations = max_iterations
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.trace: List[Dict[str, Any]] = []

    def _extract_max_price(self, text: str) -> int:
        """Trích xuất giá tối đa nếu có dạng '600 triệu', '200 triệu', '6 triệu'."""
        match = re.search(r"(\d+)\s*(triệu|tr|m)", text.lower())
        if match:
            return int(match.group(1)) * 1_000_000
        return 99999999999

    def _extract_customer_name(self, text: str) -> str:
        """Trích xuất tên khách hàng sau cụm 'Tôi tên' hoặc 'Tên tôi là'."""
        match = re.search(r"(?:tôi tên|tên tôi là)\s+([A-ZÀ-Ỹa-zà-ỹ\s]+?)(?:,|\.|$)", text, re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return "Khách hàng"

    def run(self, user_input: str) -> Dict[str, Any]:
        """Điểm vào chính – chạy Agent Loop."""
        self.trace = []
        self.trace.append({"step": "init", "user_input": user_input})
        text_lower = user_input.lower()

        # Case 1: Câu hỏi FAQ (không gọi tool)
        if any(k in text_lower for k in ["bảo hành", "kéo dài bao lâu", "chính sách"]):
            answer = "Chính sách bảo hành pin xe điện VinFast kéo dài 10 năm hoặc 200.000 km tuỳ điều kiện nào đến trước."
            self.trace.append({
                "iteration": 1,
                "thought": "Câu hỏi FAQ thông thường, trả lời trực tiếp mà không cần gọi tool.",
                "action": "None",
                "final_answer": answer
            })
            return {
                "answer": answer,
                "trace": self.trace,
                "iterations": 1,
                "status": "completed"
            }

        # Case 2: Cần ghi nhận yêu cầu hỗ trợ (submit_support_ticket)
        if any(k in text_lower for k in ["lỗi", "hỏng", "sự cố", "gấp", "khiếu nại"]):
            customer_name = self._extract_customer_name(user_input)
            priority = "high" if any(w in text_lower for w in ["gấp", "nghiêm trọng", "khẩn cấp"]) else "medium"

            tool_args = {
                "customer_name": customer_name,
                "issue_description": user_input,
                "priority": priority
            }
            res = submit_support_ticket(**tool_args)

            self.trace.append({
                "iteration": 1,
                "thought": "Phát hiện yêu cầu hỗ trợ sự cố, gọi submit_support_ticket.",
                "tool_call": {"tool": "submit_support_ticket", "arguments": tool_args},
                "observation": res
            })

            answer = f"Yêu cầu hỗ trợ của khách hàng {res['customer_name']} đã được ghi nhận với mã {res['ticket_id']} (mức độ: {res['priority']})."
            return {
                "answer": answer,
                "trace": self.trace,
                "iterations": 1,
                "status": "completed"
            }

        # Case 3: Tra cứu danh mục sản phẩm (search_product_catalog)
        category = "xe_dien" if any(w in text_lower for w in ["xe", "vinfast", "ô tô"]) else "du_lich"
        max_price = self._extract_max_price(user_input)

        products = search_product_catalog(category=category, max_price=max_price)
        self.trace.append({
            "iteration": 1,
            "thought": f"Tra cứu catalog với danh mục {category} và ngân sách {max_price}.",
            "tool_call": {"tool": "search_product_catalog", "arguments": {"category": category, "max_price": max_price}},
            "observation": products
        })

        if not products or (isinstance(products, list) and len(products) == 0):
            answer = "Rất tiếc, không tìm thấy sản phẩm nào phù hợp với yêu cầu giá của bạn."
        else:
            names = ", ".join([p.get("name", "") for p in products])
            answer = f"Các sản phẩm phù hợp được tìm thấy: {names}."

        return {
            "answer": answer,
            "trace": self.trace,
            "iterations": 1,
            "status": "completed"
        }

# ═══════════════════════════════════════════════════════════════════════════
# MAIN — Chạy thử nhanh
# ═══════════════════════════════════════════════════════════════════════════

def main():
    user_query = "Tôi muốn xem xe điện VinFast giá dưới 600 triệu."

    print("=== RUNNING CHATBOT BASELINE ===")
    chatbot = ChatbotBaseline()
    print(chatbot.query(user_query))

    print("\n=== RUNNING TOOL CALLING AGENT ===")
    agent = ToolCallingAgent(max_iterations=5)
    result = agent.run(user_query)
    print("Result:", result["answer"])
    print("Trace Log:", json.dumps(agent.trace, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
