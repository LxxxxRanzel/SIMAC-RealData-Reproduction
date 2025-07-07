import matplotlib.pyplot as plt
import torch
import torch.fft
import seaborn as sns
import math
font_size = 36
sns.set(font_scale=2)
sns.set_style('white')
plt.rcParams.update({'font.size':font_size})

class ModulatorDemodulator:
    def __init__(self, m_type="QPSK", device="cuda"):
        self.m_type = m_type
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

    def modulate(self, tensor):
        bit_stream = tensor.view(-1)

        if self.m_type == "BPSK":
            # BPSK调制映射：0 -> -1, 1 -> 1
            modulated_symbols = 2 * bit_stream - 1 # (512,) (4,1)

        elif self.m_type == "QPSK":
            # 将比特分组为符号，每组2比特
            reshaped_bits = bit_stream.view(-1, 2)
            # QPSK调制映射
            mapping = {
                (0, 0): 1 + 1j,
                (0, 1): 1 - 1j,
                (1, 0): -1 - 1j,
                (1, 1): -1 + 1j
            }
            mapping_tensor = torch.tensor(
                [mapping[(0, 0)], mapping[(0, 1)], mapping[(1, 0)], mapping[(1, 1)]],
                dtype=torch.cfloat, device=self.device
            )
            indices = reshaped_bits[:, 0] * 2 + reshaped_bits[:, 1]
            indices = indices.long()  # 确保索引为整数类型
            modulated_symbols = mapping_tensor[indices]

        elif self.m_type == "8PSK":
            # 将比特分组为符号，每组3比特
            reshaped_bits = bit_stream.view(-1, 3)
            # 8PSK调制映射
            phase_angles = torch.arange(0, 8, device=self.device) * (torch.pi / 4)
            mapping_tensor = torch.exp(1j * phase_angles)
            indices = reshaped_bits[:, 0] * 4 + reshaped_bits[:, 1] * 2 + reshaped_bits[:, 2]
            indices = indices.long()  # 确保索引为整数类型
            modulated_symbols = mapping_tensor[indices]

        elif self.m_type == "16QAM":
            # 将比特分组为符号，每组4比特
            reshaped_bits = bit_stream.view(-1, 4)
            # 16QAM调制映射
            I_values = torch.tensor([-3, -1, 1, 3], device=self.device)
            Q_values = torch.tensor([-3, -1, 1, 3], device=self.device)
            I_indices = reshaped_bits[:, 0] * 2 + reshaped_bits[:, 1]
            Q_indices = reshaped_bits[:, 2] * 2 + reshaped_bits[:, 3]
            I_indices = I_indices.long()  # 确保索引为整数类型
            Q_indices = Q_indices.long()  # 确保索引为整数类型
            modulated_symbols = I_values[I_indices] + 1j * Q_values[Q_indices]

        return modulated_symbols

    def demodulate(self, symbols, shape):
        # symbols = torch.tensor(symbols, dtype=torch.cfloat, device=self.device)
        if self.m_type == "BPSK":
            demapped_bits = (symbols.real >= 0).int()

        elif self.m_type == "QPSK":
            demapped_bits = torch.empty((symbols.size(0), 2), dtype=torch.int8, device=self.device)
            demapped_bits[:, 0] = (symbols.real < 0).int()
            demapped_bits[:, 1] = (symbols.imag < 0).int()

        elif self.m_type == "8PSK":
            phase = torch.angle(symbols)
            phase[phase < 0] += 2 * torch.pi  # 确保相位为正
            indices = (torch.round(phase / (torch.pi / 4)) % 8).long()
            demapped_bits = torch.stack([
                (indices & 4) >> 2,
                (indices & 2) >> 1,
                (indices & 1)
            ], dim=-1)

        elif self.m_type == "16QAM":
            demapped_bits = torch.empty((symbols.size(0), 4), dtype=torch.int8, device=self.device)
            I_values = torch.tensor([-3, -1, 1, 3], device=self.device)
            Q_values = torch.tensor([-3, -1, 1, 3], device=self.device)
            I_indices = torch.argmin(torch.abs(symbols.real.unsqueeze(-1) - I_values), dim=-1)
            Q_indices = torch.argmin(torch.abs(symbols.imag.unsqueeze(-1) - Q_values), dim=-1)
            demapped_bits[:, 0] = (I_indices & 2) >> 1
            demapped_bits[:, 1] = (I_indices & 1)
            demapped_bits[:, 2] = (Q_indices & 2) >> 1
            demapped_bits[:, 3] = (Q_indices & 1)
        demapped_bits = demapped_bits.flatten()
        tensor = demapped_bits.view(shape)
        return tensor

    def normalize_complex_tensor(self,tensor):
        # 分别获取实部和虚部的最大绝对值
        real_max = torch.max(torch.abs(tensor.real))
        imag_max = torch.max(torch.abs(tensor.imag))
        # 找到实部和虚部的最大值
        max_value = max(real_max, imag_max)
        # 避免除以零
        if max_value == 0:
            return tensor
        # 归一化到 [-1, 1]
        normalized_tensor = tensor / max_value
        return normalized_tensor

    def Constellation_draw(self, modulated_symbols, received_symbols,save_path=None):
        plt.figure(figsize=(12, 6))
        modulated_symbols = self.normalize_complex_tensor(modulated_symbols)
        received_symbols = self.normalize_complex_tensor(received_symbols)
        # 原始信号的星座图
        plt.subplot(1, 2, 1)
        plt.scatter(modulated_symbols.cpu().numpy().real, modulated_symbols.cpu().numpy().imag, color='blue', s=10)
        plt.title('Modulated Constellation Diagram')
        plt.xlabel('In-Phase (I)')
        plt.ylabel('Quadrature (Q)')
        plt.grid()
        plt.axis('equal')
        # plt.xlim(-2, 2)
        # plt.ylim(-2, 2)
        plt.axhline(0, color='black', linewidth=0.5, linestyle='--')
        plt.axvline(0, color='black', linewidth=0.5, linestyle='--')

        # 接收信号的星座图
        plt.subplot(1, 2, 2)
        plt.scatter(received_symbols.cpu().numpy().real, received_symbols.cpu().numpy().imag, color='red', s=10)
        plt.title('Received Constellation Diagram')
        plt.xlabel('In-Phase (I)')
        plt.ylabel('Quadrature (Q)')
        plt.grid()
        plt.axis('equal')
        # plt.xlim(-2, 2)
        # plt.ylim(-2, 2)
        plt.axhline(0, color='black', linewidth=0.5, linestyle='--')
        plt.axvline(0, color='black', linewidth=0.5, linestyle='--')
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path,format='pdf',bbox_inches='tight', pad_inches=0.1)
        else:
            plt.show()
        plt.close()

        # plt.show()

class PhysicalChannel:
    def __init__(self, c_type="AWGN", device="cuda"):
        self.c_type = c_type
        self.device = device

    def __call__(self, signal, Eb_N0_dB, K=5):
        # 将信号移动到指定设备
        signal = signal.to(self.device)
        # 计算信号功率（按信号的最后两维平均）
        signal_power = torch.mean(torch.abs(signal) ** 2)

        # 计算噪声功率
        N0 = signal_power / (10 ** (Eb_N0_dB / 10))  # 噪声功率

        # 生成 AWGN 噪声
        noise_real = torch.randn_like(signal, dtype=torch.float32, device=self.device)
        noise_imag = torch.randn_like(signal, dtype=torch.float32, device=self.device)
        noise = torch.sqrt(N0 / 2) * (noise_real + 1j * noise_imag)  # 复数噪声

        if self.c_type == "AWGN":
            # AWGN 信道
            return signal + noise
        elif self.c_type == "Rician":
            # Rician 信道
            s = torch.sqrt(torch.tensor(2.0, device=self.device)) * (K / (K + 1))
            noise_power = 1 / (K + 1)

            # 生成 Rician 衰落
            h_real = math.sqrt(noise_power / 2) * torch.randn_like(signal, dtype=torch.float32, device=self.device)
            h_imag = math.sqrt(noise_power / 2) * torch.randn_like(signal, dtype=torch.float32, device=self.device)
            h = (h_real + 1j * h_imag) + s

            # 缩放因子
            scaling_factor = torch.sqrt((10 ** (Eb_N0_dB / 10)) / signal_power)
            return signal * scaling_factor * h + noise
        elif self.c_type == "Rayleigh":

            # Rayleigh 信道
            h_real = torch.randn_like(signal, dtype=torch.float32, device=self.device)
            h_imag = torch.randn_like(signal, dtype=torch.float32, device=self.device)
            h = (h_real + 1j * h_imag) / torch.sqrt(torch.tensor(2.0, device=self.device))

            # 缩放因子
            scaling_factor = torch.sqrt((10 ** (Eb_N0_dB / 10)) / signal_power)
            return 10*signal * scaling_factor * h + noise

