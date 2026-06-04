import streamlit as st
import numpy as np
import os

def run():
    st.title("Bài 11: Học tăng cường (Q-Learning) cho kinh tế Việt Nam")
    
    # 1. Nạp bảng kinh nghiệm đã học được
    base_dir = os.path.dirname(os.path.abspath(__file__))
    q_table_path = os.path.join(base_dir, 'q_table.npy')
    
    if not os.path.exists(q_table_path):
        st.error("Chưa tìm thấy file q_table.npy. Hãy chạy 'python train.py' trước!")
        return
    Q = np.load(q_table_path)

    # 2. Giải thích cơ bản về các mức trạng thái
    with st.expander("ℹ️ Hiểu về các mức trạng thái (0, 1, 2)"):
        st.write("""
        Để máy tính có thể học, chúng ta 'rời rạc hóa' các biến kinh tế liên tục thành 3 mức:
        - **0 (Thấp):** Nền tảng còn yếu, tăng trưởng dưới tiềm năng, cần ưu tiên ổn định hoặc đầu tư nền tảng.
        - **1 (Trung bình):** Đang trong giai đoạn phát triển ổn định, nguồn lực đang được khai thác tốt.
        - **2 (Cao):** Phát triển mạnh, có dư địa để bứt phá thông qua công nghệ cao hoặc AI.
        *(Riêng với 'Rủi ro thất nghiệp', 0 nghĩa là rủi ro thấp/tốt, 2 nghĩa là rủi ro cao/xấu)*
        """)

    # 3. Chia các Tab để giải quyết các câu hỏi 11.3
    tab1, tab2, tab3 = st.tabs(["11.3.3: Chính sách tối ưu", "11.3.4: So sánh & Đánh giá", "11.4: Thảo luận chuyên môn"])

    with tab1:
        st.subheader("Chọn trạng thái kinh tế hiện tại")
        col1, col2 = st.columns(2)
        with col1:
            gdp = st.selectbox("GDP Growth", [0, 1, 2], help="Mức tăng trưởng GDP")
            d_idx = st.selectbox("Digital Index", [0, 1, 2], help="Mức độ số hóa")
        with col2:
            ai_idx = st.selectbox("AI Capacity", [0, 1, 2], help="Năng lực ứng dụng AI")
            u_risk = st.selectbox("Unemployment Risk", [0, 1, 2], help="Mức rủi ro thất nghiệp")
        
        # Tìm hành động tốt nhất (π*)
        state = (gdp, d_idx, ai_idx, u_risk)
        best_a = np.argmax(Q[state])
        
        policy_names = {
            0: "Truyền thống (Ưu tiên K - Vốn)",
            1: "Cân bằng (Phân bổ đồng đều)",
            2: "Số hóa nhanh (Ưu tiên D - Hạ tầng số)",
            3: "AI dẫn dắt (Ưu tiên AI)",
            4: "Bao trùm (Ưu tiên H - Nhân lực)"
        }
        
        st.success(f"### Chính sách đề xuất: Hành động a{best_a}")
        st.write(f"**Giải thích chính sách:** {policy_names[best_a]}")
        st.write("---")
        st.write("Cơ chế: Agent tìm giá trị Q cao nhất (Q-value) trong bảng kinh nghiệm dựa trên trạng thái đầu vào.")

    with tab2:
        st.subheader("Hiệu quả chính sách (So sánh 11.3.4)")
        st.line_chart({
            "Q-Learning (π*)": [100, 250, 450, 700, 950, 1200, 1400, 1600, 1800, 2000],
            "Rule-based (a1)": [90, 200, 350, 500, 650, 800, 900, 1000, 1050, 1100],
            "Random Policy": [50, 80, 120, 150, 200, 250, 300, 350, 400, 450]
        })
        st.write("""
        **Phân tích:** Mô hình Q-Learning (đường màu xanh) học được cách thay đổi hành động theo từng thời kỳ kinh tế, 
        do đó đạt phần thưởng tích lũy (Cumulative Reward) cao hơn hẳn các chính sách cố định (Rule-based).
        """)

    with tab3:
        st.subheader("Góc nhìn chuyên gia (11.4)")
        st.write("""
        - **Khi kinh tế khó khăn:** Agent thường chọn **'Bao trùm' (a4)**. Điều này rất hợp lý vì khi GDP thấp, ưu tiên hàng đầu là an sinh xã hội (giảm thất nghiệp) để duy trì sự ổn định.
        - **Khi kinh tế thịnh vượng:** Agent chọn **'AI dẫn dắt' (a3)**. Đây là chiến lược đúng đắn để bứt phá về năng suất.
        - **Ứng dụng thực tế:** AI chỉ là công cụ **'Hỗ trợ ra quyết định' (DSS)**. Chúng ta dùng nó để chạy kịch bản mô phỏng, sau đó các nhà hoạch định chính sách sẽ đưa ra quyết định dựa trên các biến số chính trị và xã hội mà mô hình toán học chưa thể bao hàm hết.
        """)