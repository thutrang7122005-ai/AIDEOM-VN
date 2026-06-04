import numpy as np
from env import VietnamEconomyEnv
import os

# Khởi tạo môi trường
env = VietnamEconomyEnv()

# Khởi tạo bảng Q (81 trạng thái x 5 hành động)
# Mỗi chỉ số có 3 mức, 4 chỉ số -> 3^4 = 81 trạng thái
Q = np.zeros((3, 3, 3, 3, 5))

# Tham số huấn luyện
lr = 0.1       # Learning rate
gamma = 0.95   # Discount factor
eps = 1.0      # Epsilon bắt đầu
min_eps = 0.05
episodes = 10000

print("Đang huấn luyện Agent... Vui lòng chờ!")

for ep in range(episodes):
    s, _ = env.reset()
    done = False
    
    while not done:
        # Epsilon-greedy: Quyết định giữa khám phá (random) và khai thác (Q-table)
        if np.random.rand() < eps:
            a = env.action_space.sample()
        else:
            a = int(np.argmax(Q[tuple(s)]))
            
        s2, r, done, _, _ = env.step(a)
        
        # Công thức Q-learning
        Q[tuple(s) + (a,)] += lr * (r + gamma * np.max(Q[tuple(s2)]) - Q[tuple(s) + (a,)])
        
        s = s2
        
    # Giảm dần epsilon
    if eps > min_eps:
        eps -= (1.0 - min_eps) / episodes

# Lưu bảng Q đã học được
np.save('q_table.npy', Q)
print("Huấn luyện xong! Bảng chính sách đã lưu vào file 'q_table.npy'.")