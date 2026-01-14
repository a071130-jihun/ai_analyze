import numpy as np
import torch


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


class TimeShift:
    def __init__(self, max_shift: int = 10, p: float = 0.5):
        self.max_shift = max_shift
        self.p = p
    
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if np.random.random() > self.p:
            return x
        shift = np.random.randint(-self.max_shift, self.max_shift + 1)
        return torch.roll(x, shifts=shift, dims=-1)


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


class Compose:
    def __init__(self, transforms: list):
        self.transforms = transforms
    
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        for t in self.transforms:
            x = t(x)
        return x


def get_train_transform():
    return Compose([
        SpecAugment(freq_mask_param=15, time_mask_param=15, n_freq_masks=2, n_time_masks=2, p=0.5),
        RandomNoise(noise_level=0.005, p=0.3),
        TimeShift(max_shift=5, p=0.3),
        RandomGain(min_gain=0.9, max_gain=1.1, p=0.3),
    ])


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
