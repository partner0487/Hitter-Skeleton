import math

def smoothing_factor(t_e, cutoff):
    """计算平滑因子"""
    r = 2 * math.pi * cutoff * t_e
    return r / (r + 1)

def exponential_smoothing(a, x, x_prev):
    """执行指数平滑"""
    return a * x + (1 - a) * x_prev

class OneEuroFilter:
    def __init__(self, dx0=0.0, min_cutoff=0.15, beta=0.3, d_cutoff=1.0):
        """
        初始化 One Euro Filter.

        参数:
            dx0 (float): 初始差分值.
            min_cutoff (float): 最小截止频率.
            beta (float): 截止频率的调节系数.
            d_cutoff (float): 差分的截止频率.
        """
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self.dx_prev = float(dx0)

    def __call__(self, x, x_prev):
        """
        执行滤波操作.

        参数:
            x (float): 当前值.
            x_prev (float): 前一个值.

        返回:
            x_hat (float): 滤波后的值.
        """
        if x_prev is None:
            return x

        # 假设时间步长为1秒，根据实际情况可以调整
        t_e = 1  

        # 计算差分的平滑因子
        a_d = smoothing_factor(t_e, self.d_cutoff)
        dx = (x - x_prev) / t_e
        dx_hat = exponential_smoothing(a_d, dx, self.dx_prev)

        # 计算动态截止频率
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)

        # 计算值的平滑因子
        a = smoothing_factor(t_e, cutoff)
        x_hat = exponential_smoothing(a, x, x_prev)

        # 更新前一个差分值
        self.dx_prev = dx_hat

        return x_hat

# 示例使用
if __name__ == "__main__":
    filter = OneEuroFilter()
    values = [1, 2, 3, 2, 1]  # 示例数据
    filtered_values = []

    prev_value = None
    for value in values:
        filtered_value = filter(value, prev_value)
        filtered_values.append(filtered_value)
        prev_value = value

    print(filtered_values)
