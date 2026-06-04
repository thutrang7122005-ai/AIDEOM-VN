import gymnasium as gym
from gymnasium import spaces
import numpy as np

class VietnamEconomyEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.MultiDiscrete([3, 3, 3, 3])
        self.allocation = {
            0: [0.70, 0.10, 0.10, 0.10], 1: [0.40, 0.25, 0.15, 0.20],
            2: [0.25, 0.45, 0.15, 0.15], 3: [0.20, 0.20, 0.45, 0.15],
            4: [0.30, 0.20, 0.10, 0.40]
        }
        self.w = np.array([0.40, 0.25, 0.20, 0.15])
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = np.array([1, 1, 0, 1]) # [GDP_level, D_level, AI_level, U_level]
        self.t = 0
        self.K, self.D, self.AI, self.H = 27500.0, 20.3, 86.0, 30.0
        return self.state, {}

    def discretize(self, val, thresholds):
        """Chuyển giá trị thực sang mức 0, 1, 2"""
        if val < thresholds[0]: return 0
        elif val < thresholds[1]: return 1
        return 2

    def step(self, action):
        # 1. Lưu giá trị trước khi cập nhật để tính Δ (delta)
        Y_prev = (self.K**0.33) * (54.0**0.42) * (self.D**0.10) * (self.AI**0.08) * (self.H**0.07)
        
        # 2. Cập nhật các biến kinh tế
        a = self.allocation[action]
        budget = 1000
        self.K += a[0]*budget; self.D += a[1]*budget/100
        self.AI += a[2]*budget/20; self.H += a[3]*budget/200
        
        # 3. Tính toán Y và thay đổi
        Y_now = (self.K**0.33) * (54.0**0.42) * (self.D**0.10) * (self.AI**0.08) * (self.H**0.07)
        delta_GDP = (Y_now - Y_prev) / Y_prev
        
        # 4. Giả định các chỉ số rủi ro/thất nghiệp (theo đề bài)
        unemploy_risk = 0.1 / (1 + self.AI/100) # Ví dụ: AI cao làm giảm U
        cyber_risk = 0.05 * (self.D / 100)
        emission = 0.02 * (self.K / 30000)
        
        # 5. Công thức Reward (w = [0.40, 0.25, 0.20, 0.15])
        reward = (self.w[0] * delta_GDP) - (self.w[1] * unemploy_risk) - \
                 (self.w[2] * cyber_risk) - (self.w[3] * emission)
        
        # 6. Cập nhật state rời rạc (Ví dụ ngưỡng cho K, D, AI, H)
        self.state = np.array([
            self.discretize(self.K, [25000, 30000]),
            self.discretize(self.D, [25, 50]),
            self.discretize(self.AI, [100, 200]),
            self.discretize(self.H, [40, 60])
        ])
        
        self.t += 1
        return self.state, float(reward), self.t >= 10, False, {}