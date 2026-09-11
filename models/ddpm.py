import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class DDPM(nn.Module):
    def __init__(
        self,
        generator,
        num_train_steps=1000,
        beta_start: float = 0.00085,
        beta_end: float = 0.012,
        schedule='cosine',
        cond_dropout_prob=0.1,
    ):
        super().__init__()
        if schedule == 'cosine':
            # 余弦调度：初始加噪慢，保留更多结构
            self.betas = self._cosine_beta_schedule(num_train_steps)
        else:
            # 原有的线性调度
            self.betas = torch.linspace(beta_start**0.5, beta_end**0.5, num_train_steps)**2
        self.alphas = 1.0 - self.betas  # 计算 alphas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)  # 计算 alpha 的累积积
        self.one = torch.tensor(1.0)

        self.generator = generator  # unet
        self.cond_dropout_prob = cond_dropout_prob
        
        self.num_train_timesteps = num_train_steps
        self.timesteps = torch.from_numpy(np.arange(0, num_train_steps)[::-1].copy())
        

    def _cosine_beta_schedule(self, timesteps, s=0.008):
        """余弦调度，来自Improved DDPM论文"""
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0, 0.999)
    
    def add_noise(
        self,
        original_samples: torch.FloatTensor,
        timesteps: torch.IntTensor,
    ) -> torch.FloatTensor:
        alphas_cumprod = self.alphas_cumprod.to(
            device=original_samples.device, dtype=original_samples.dtype
        )
        # timesteps = timesteps.to(original_samples.device)

        sqrt_alpha_prod = alphas_cumprod[timesteps] ** 0.5
        sqrt_alpha_prod = sqrt_alpha_prod.flatten()
        while len(sqrt_alpha_prod.shape) < len(original_samples.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)

        sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[timesteps]) ** 0.5
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
        while len(sqrt_one_minus_alpha_prod.shape) < len(original_samples.shape):
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

        noise = torch.randn(
            original_samples.shape,
            device=original_samples.device,
            dtype=original_samples.dtype,
        )
        noisy_samples = (
            sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise
        )
        return noisy_samples, noise

    def forward(self, x, timesteps, solar_wind):
        #训练时随机丢弃条件（分类器无关引导的关键）
        # if self.training and torch.rand(1) < self.cond_dropout_prob:
        #     # 使用全零mask（表示所有区域都需要生成）
        #     mask = torch.zeros_like(mask)
        return self.generator(x, timesteps, solar_wind)

    def set_inference_timesteps(self, num_inference_steps=50):
        self.num_inference_steps = num_inference_steps

        step_ratio = self.num_train_timesteps // self.num_inference_steps
        self.step_ratio = step_ratio
        timesteps = (
            (np.arange(0, num_inference_steps) * step_ratio)
            .round()[::-1]
            .copy()
            .astype(np.int64)
        )
        self.timesteps = torch.from_numpy(timesteps)

    # def _get_previous_timestep(self, timestep: int) -> int:
    #     prev_t = timestep - self.num_train_timesteps // self.num_inference_steps
    #     return prev_t

    def _get_variance(self, timestep: int) -> torch.Tensor:
        # prev_t = self._get_previous_timestep(timestep)
        prev_t = timestep - 1
        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.alphas_cumprod[prev_t] if prev_t >= 0 else self.one
        current_beta_t = 1 - alpha_prod_t / alpha_prod_t_prev

        variance = (1 - alpha_prod_t_prev) / (1 - alpha_prod_t) * current_beta_t

        variance = torch.clamp(variance, min=1e-20)

        return variance

    def get_inference_schedule(
        self,
        t_T=250,
        n_sample=1,
        jump_len=10,
        jump_n_sample=10,
        start_resampling=100000000,
    ):
        jumps = {}
        for j in range(0, t_T - jump_len, jump_len):
            jumps[j] = jump_n_sample - 1

        t = t_T
        ts = []

        while t >= 1:
            t = t - 1
            ts.append(t)
            if t + 1 < t_T - 1 and t <= start_resampling:
                for _ in range(n_sample - 1):
                    t = t + 1
                    ts.append(t)
                    if t >= 0:
                        t = t - 1
                        ts.append(t)

            if jumps.get(t, 0) > 0 and t <= start_resampling - jump_len:
                jumps[t] = jumps[t] - 1
                for _ in range(jump_len):
                    t = t + 1
                    ts.append(t)
        ts.append(-1)
        return ts


    def step(self, timestep, xt_image, x0_image, mask, solar_point):
        # 🔧 修复1：确保timestep是整数类型
        if timestep.dtype != torch.long:
            timestep = timestep.long()
        # 方法1：直接使用timestep[0]索引（推荐）
        t = timestep[0].item() if timestep.dim() > 0 else timestep.item()

        if t == 0:
            # 最后一次去噪，直接合并
            # soft_mask_final = self.get_soft_mask(mask, t, use_time_adaptive=True)
            return mask * x0_image + (1 - mask) * xt_image
            #return xt_image
        prev_t = t - 1
        
        # 确保prev_t有效
        if prev_t < 0:
            prev_t = 0
        
        alpha_prod_t = self.alphas_cumprod[t]  
        alpha_prod_t_prev = self.alphas_cumprod[prev_t] if prev_t >= 0 else self.one
        
        # 剩余代码保持不变...
        current_alpha_t = alpha_prod_t / alpha_prod_t_prev
        current_beta_t = 1 - current_alpha_t
        
        # 重噪声化：获取当前时间步的加噪已知区域
        x_known_t, _ = self.add_noise(x0_image, timestep)  #这里传递张量
        
        # 构造当前输入
        #soft_mask_current = self.get_soft_mask(mask, t, use_time_adaptive=True)
        x_current_t = mask * x_known_t + (1 - mask) * xt_image
        
        # 模型预测
        epsilon_theta = self(x_current_t, timestep,solar_point)
        
        # 无条件预测
        # zero_mask = torch.zeros_like(mask)
        # zero_x0 = torch.zeros_like(x0_image)
        # epsilon_uncond = self(x_current_t, timestep, zero_mask, zero_x0)
        
        # 引导融合
        #epsilon_theta = epsilon_uncond + guidance_scale * (epsilon_cond - epsilon_uncond)
        
        # DDPM标准去噪步骤
        variance = 0
        if t > 0:
            variance = self._get_variance(t) ** 0.5
        
        mean = (x_current_t - current_beta_t / torch.sqrt(1 - alpha_prod_t) * epsilon_theta) / torch.sqrt(current_alpha_t)
        
        # 噪声
        z = torch.randn_like(xt_image) if t > 0 else torch.zeros_like(xt_image)
        z = z.to(xt_image.device)
        
        x_pred_t_minus_1 = mean + variance * z
        
        # 获取t-1时刻的加噪已知区域
        if t > 1:
            prev_t_tensor = torch.tensor([prev_t] * xt_image.shape[0], 
                                        device=xt_image.device, 
                                        dtype=torch.long)
            x_known_t_minus_1, _ = self.add_noise(x0_image, prev_t_tensor)
        else:
            # t=1时，t-1=0，直接用原图
            x_known_t_minus_1 = x0_image
        
        # 合并结果
        x_merged = mask * x_known_t_minus_1 + (1 - mask) * x_pred_t_minus_1
        
        return x_merged

    def sample(self, image, mask, solar_point,num_inference_steps=250, n_sample=1, j=10, r=10):
        times = self.get_inference_schedule(
            t_T=num_inference_steps, n_sample=n_sample, jump_len=j, jump_n_sample=r
        )  # 获得时间步表

        device = image.device
        x_t = torch.randn_like(image, device=device, dtype=torch.float32)
         # 2. 获取初始时间步T（最大的时间步）
        T = times[0]
        
        # 3. 计算T时刻的加噪已知区域
        x_known_T, _ = self.add_noise(image, T)
        
        # 4. 构造初始x_t：已知区域用T时刻加噪，未知区域用纯噪声
        #x_t = mask * x_known_T + (1 - mask) * x_t
        x_t = x_known_T
        image = image.to(device, dtype=torch.float32)  # 原始输入图像
        # x_0=image.clone()
        for t_last, t_cur in zip(times[:-1], times[1:]):
            if t_cur < t_last:  # 反向
                t_tensor = torch.tensor(
                    [t_last] * image.shape[0], device=device, dtype=torch.long
                )
                x_t = self.step(t_tensor, x_t, image, mask, solar_point)

            else:  # 前向
                # 重噪声化跳转
                new_t = t_last + 1
                # 对已知区域重新加噪
                x_known_new, _ = self.add_noise(image, new_t)
                # 对当前图像加噪
                x_t_noised = self.addNoiseFromXt(x_t, new_t)
                # 合并
                x_t = mask * x_known_new + (1 - mask) * x_t_noised
                #x_t = x_t_noised
        # print(t_last, t_cur)  # 查看最后一步时前向还是反向
        x_t = mask * image + (1 - mask) * x_t  # 强制保留已知部分不变
        return x_t

    def addNoiseFromXt(self, x_pre, t):
        pre_t = t - 1
        alpha_prod_t = self.alphas_cumprod[t]
        alpha_prod_t_prev = self.alphas_cumprod[pre_t] if pre_t >= 0 else self.one
        current_alpha_t = alpha_prod_t / alpha_prod_t_prev
        current_beta_t = 1 - current_alpha_t
        noise = torch.randn(
            x_pre.shape,
            device=x_pre.device,
            dtype=x_pre.dtype,
        )
        x_t = torch.sqrt(1 - current_beta_t) * x_pre + (current_beta_t**0.5) * noise

        return x_t
