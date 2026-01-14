import numpy as np
import torch
import torch.nn.functional as F


class SpecAugment:
    def __init__(
        self,
        freq_mask_param: int = 20,
        time_mask_param: int = 20,
        n_freq_masks: int = 2,
        n_time_masks: int = 2,
        p: float = 0.5
    ):
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param
        self.n_freq_masks = n_freq_masks
        self.n_time_masks = n_time_masks
        self.p = p
    
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if np.random.random() > self.p:
            return x
        
        x = x.clone()
        _, n_freq, n_time = x.shape
        
        for _ in range(self.n_freq_masks):
            f = np.random.randint(0, self.freq_mask_param)
            f0 = np.random.randint(0, max(1, n_freq - f))
            x[:, f0:f0+f, :] = 0
        
        for _ in range(self.n_time_masks):
            t = np.random.randint(0, self.time_mask_param)
            t0 = np.random.randint(0, max(1, n_time - t))
            x[:, :, t0:t0+t] = 0
        
        return x


class RandomNoise:
    def __init__(self, noise_level: float = 0.01, p: float = 0.5):
        self.noise_level = noise_level
        self.p = p
    
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if np.random.random() > self.p:
            return x
        noise = torch.randn_like(x) * self.noise_level
        return x + noise


class SNRNoise:
    def __init__(self, snr_min: float = -10, snr_max: float = 10, p: float = 0.5):
        self.snr_min = snr_min
        self.snr_max = snr_max
        self.p = p
    
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if np.random.random() > self.p:
            return x
        
        snr_db = np.random.uniform(self.snr_min, self.snr_max)
        snr_linear = 10 ** (snr_db / 10)
        
        signal_power = torch.mean(x ** 2)
        noise_power = signal_power / snr_linear
        noise = torch.randn_like(x) * torch.sqrt(noise_power)
        
        return x + noise


class TimeShift:
    def __init__(self, max_shift: int = 10, p: float = 0.5):
        self.max_shift = max_shift
        self.p = p
    
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if np.random.random() > self.p:
            return x
        shift = np.random.randint(-self.max_shift, self.max_shift + 1)
        return torch.roll(x, shifts=shift, dims=-1)


class FrequencyShift:
    def __init__(self, max_shift: int = 5, p: float = 0.5):
        self.max_shift = max_shift
        self.p = p
    
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if np.random.random() > self.p:
            return x
        shift = np.random.randint(-self.max_shift, self.max_shift + 1)
        return torch.roll(x, shifts=shift, dims=-2)


class TimeStretch:
    def __init__(self, min_rate: float = 0.8, max_rate: float = 1.2, p: float = 0.5):
        self.min_rate = min_rate
        self.max_rate = max_rate
        self.p = p
    
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if np.random.random() > self.p:
            return x
        
        rate = np.random.uniform(self.min_rate, self.max_rate)
        _, n_freq, n_time = x.shape
        new_time = int(n_time * rate)
        
        x_stretched = F.interpolate(
            x.unsqueeze(0), 
            size=(n_freq, new_time), 
            mode='bilinear', 
            align_corners=False
        ).squeeze(0)
        
        if new_time > n_time:
            start = (new_time - n_time) // 2
            x_stretched = x_stretched[:, :, start:start + n_time]
        else:
            pad_left = (n_time - new_time) // 2
            pad_right = n_time - new_time - pad_left
            x_stretched = F.pad(x_stretched, (pad_left, pad_right), mode='constant', value=0)
        
        return x_stretched


class RandomCutout:
    def __init__(self, n_holes: int = 3, max_h: int = 20, max_w: int = 20, p: float = 0.5):
        self.n_holes = n_holes
        self.max_h = max_h
        self.max_w = max_w
        self.p = p
    
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if np.random.random() > self.p:
            return x
        
        x = x.clone()
        _, n_freq, n_time = x.shape
        
        for _ in range(self.n_holes):
            h = np.random.randint(1, self.max_h + 1)
            w = np.random.randint(1, self.max_w + 1)
            y = np.random.randint(0, max(1, n_freq - h))
            t = np.random.randint(0, max(1, n_time - w))
            x[:, y:y+h, t:t+w] = 0
        
        return x


class RandomGain:
    def __init__(self, min_gain: float = 0.8, max_gain: float = 1.2, p: float = 0.5):
        self.min_gain = min_gain
        self.max_gain = max_gain
        self.p = p
    
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if np.random.random() > self.p:
            return x
        gain = np.random.uniform(self.min_gain, self.max_gain)
        return x * gain


class RandomErasing:
    def __init__(self, p: float = 0.5, scale: tuple = (0.02, 0.2), ratio: tuple = (0.3, 3.3)):
        self.p = p
        self.scale = scale
        self.ratio = ratio
    
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if np.random.random() > self.p:
            return x
        
        x = x.clone()
        _, h, w = x.shape
        area = h * w
        
        target_area = np.random.uniform(self.scale[0], self.scale[1]) * area
        aspect_ratio = np.random.uniform(self.ratio[0], self.ratio[1])
        
        erase_h = int(round(np.sqrt(target_area * aspect_ratio)))
        erase_w = int(round(np.sqrt(target_area / aspect_ratio)))
        
        if erase_h < h and erase_w < w:
            y = np.random.randint(0, h - erase_h)
            t = np.random.randint(0, w - erase_w)
            x[:, y:y+erase_h, t:t+erase_w] = torch.randn(1, erase_h, erase_w) * 0.1
        
        return x


class Compose:
    def __init__(self, transforms: list):
        self.transforms = transforms
    
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        for t in self.transforms:
            x = t(x)
        return x


class RandomApply:
    def __init__(self, transforms: list, p: float = 0.5):
        self.transforms = transforms
        self.p = p
    
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if np.random.random() < self.p:
            for t in self.transforms:
                x = t(x)
        return x


def get_train_transform(strength: str = "medium"):
    if strength == "light":
        return Compose([
            SpecAugment(freq_mask_param=10, time_mask_param=10, n_freq_masks=1, n_time_masks=1, p=0.3),
            RandomNoise(noise_level=0.005, p=0.2),
            RandomGain(min_gain=0.95, max_gain=1.05, p=0.2),
        ])
    
    elif strength == "medium":
        return Compose([
            SpecAugment(freq_mask_param=20, time_mask_param=25, n_freq_masks=2, n_time_masks=2, p=0.5),
            SNRNoise(snr_min=5, snr_max=20, p=0.3),
            TimeShift(max_shift=10, p=0.3),
            FrequencyShift(max_shift=3, p=0.3),
            RandomGain(min_gain=0.85, max_gain=1.15, p=0.3),
        ])
    
    elif strength == "strong":
        return Compose([
            SpecAugment(freq_mask_param=30, time_mask_param=35, n_freq_masks=3, n_time_masks=3, p=0.7),
            SNRNoise(snr_min=-5, snr_max=15, p=0.5),
            TimeStretch(min_rate=0.85, max_rate=1.15, p=0.4),
            TimeShift(max_shift=15, p=0.4),
            FrequencyShift(max_shift=5, p=0.4),
            RandomCutout(n_holes=2, max_h=15, max_w=15, p=0.4),
            RandomGain(min_gain=0.7, max_gain=1.3, p=0.4),
        ])
    
    elif strength == "aggressive":
        return Compose([
            SpecAugment(freq_mask_param=40, time_mask_param=45, n_freq_masks=4, n_time_masks=4, p=0.8),
            SNRNoise(snr_min=-10, snr_max=10, p=0.6),
            TimeStretch(min_rate=0.8, max_rate=1.2, p=0.5),
            TimeShift(max_shift=20, p=0.5),
            FrequencyShift(max_shift=8, p=0.5),
            RandomCutout(n_holes=3, max_h=20, max_w=25, p=0.5),
            RandomErasing(p=0.4, scale=(0.05, 0.25)),
            RandomGain(min_gain=0.6, max_gain=1.4, p=0.5),
        ])
    
    return get_train_transform("medium")


def mixup_data(x, y, alpha=0.2):
    """Mixup 증강: 두 샘플을 섞어서 새로운 샘플 생성"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Mixup용 손실 함수"""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
