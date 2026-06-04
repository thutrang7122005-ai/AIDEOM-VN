import os

# Đảm bảo thư mục 'modules' tồn tại
if not os.path.exists('modules'):
    os.makedirs('modules')

# Tạo file __init__.py trong modules
with open('modules/__init__.py', 'w') as f:
    pass

# Tạo 12 file bai1.py đến bai12.py trong thư mục modules
for i in range(1, 13):
    file_path = f'modules/bai{i}.py'
    with open(file_path, 'w') as f:
        f.write("import streamlit as st\n\n")
        f.write("def run():\n")
        f.write("    st.header('Đang xây dựng...')\n")
    print(f"Đã tạo: {file_path}")