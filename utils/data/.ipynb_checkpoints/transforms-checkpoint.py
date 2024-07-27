import numpy as np
import torch
from MRAugment.mraugment.data_augment import DataAugmentor

def to_tensor(data):
    """
    Convert numpy array to PyTorch tensor. For complex arrays, the real and imaginary parts
    are stacked along the last dimension.
    Args:
        data (np.array): Input numpy array
    Returns:
        torch.Tensor: PyTorch version of data
    """
    return torch.from_numpy(data)

class DataTransform:
    def __init__(self, isforward, max_key, augmentor=None):
        self.isforward = isforward
        self.max_key = max_key
        self.augmentor = augmentor  # DataAugmentor 추가
    """
    def __call__(self, mask, input, target, attrs, fname, slice):
        if not self.isforward:
            target = to_tensor(target)
            maximum = attrs[self.max_key]
        else:
            target = -1
            maximum = -1
            
        kspace = to_tensor(input)
        kspace = torch.stack((kspace.real, kspace.imag), dim=-1)
        
        if self.augmentor is not None and self.augmentor.schedule_p() > 0.0:
            kspace, target = self.augmentor(kspace, target.shape if target is not None else None)
        
        kspace = kspace * to_tensor(mask)
        #kspace = torch.stack((kspace.real, kspace.imag), dim=-1)
        mask = torch.from_numpy(mask.reshape(1, 1, kspace.shape[-2], 1).astype(np.float32)).byte()
        #print(f"Augmented kspace shape or: {kspace.shape}")
        
        return mask, kspace, target, maximum, fname, slice
    """
    
    def __call__(self, mask, input, target, attrs, fname, slice):
        if not self.isforward:
            target = to_tensor(target)
            maximum = attrs[self.max_key]
        else:
            target = -1
            maximum = -1
        
        input_tensor = to_tensor(input)
        
        # Augmentation 전 복소수 차원 분리
        input_tensor = torch.stack((input_tensor.real, input_tensor.imag), dim=-1)
        
        # Data Augmentation 수행
        if self.augmentor is not None and self.augmentor.schedule_p() > 0.0:
            input_tensor, target = self.augmentor(input_tensor, target.shape if target is not None else None)
        
        # Augmentation 후 복소수 차원 합침
        input_tensor = torch.complex(input_tensor[..., 0], input_tensor[..., 1])
        
        # Masking 적용
        kspace = input_tensor * to_tensor(mask)
        kspace = torch.stack((kspace.real, kspace.imag), dim=-1)
        mask = torch.from_numpy(mask.reshape(1, 1, kspace.shape[-2], 1).astype(np.float32)).byte()
        
        return mask, kspace, target, maximum, fname, slice

# DataTransform: augmentor 추가 완료 (0727, Reorder DA and Masking)