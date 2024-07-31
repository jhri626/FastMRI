"""
Copyright (c) Facebook, Inc. and its affiliates.
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SSIMLoss(nn.Module):
    """
    SSIM loss module.
    """

    def __init__(self, win_size: int = 7, k1: float = 0.01, k2: float = 0.03):
        """
        Args:
            win_size: Window size for SSIM calculation.
            k1: k1 parameter for SSIM calculation.
            k2: k2 parameter for SSIM calculation.
        """
        super().__init__()
        self.win_size = win_size
        self.k1, self.k2 = k1, k2
        self.register_buffer("w", torch.ones(1, 1, win_size, win_size) / win_size ** 2)
        NP = win_size ** 2
        self.cov_norm = NP / (NP - 1)

    def forward(self, X, Y, data_range):
        print('\n w:',self.w)
        X = X.unsqueeze(1)
        Y = Y.unsqueeze(1)
        data_range = data_range[:, None, None, None]
        C1 = (self.k1 * data_range) ** 2
        C2 = (self.k2 * data_range) ** 2
        ux = F.conv2d(X, self.w)
        uy = F.conv2d(Y, self.w)
        uxx = F.conv2d(X * X, self.w)
        uyy = F.conv2d(Y * Y, self.w)
        uxy = F.conv2d(X * Y, self.w)
        vx = self.cov_norm * (uxx - ux * ux)
        vy = self.cov_norm * (uyy - uy * uy)
        vxy = self.cov_norm * (uxy - ux * uy)
        A1, A2, B1, B2 = (
            2 * ux * uy + C1,
            2 * vxy + C2,
            ux ** 2 + uy ** 2 + C1,
            vx + vy + C2,
        )
        D = B1 * B2
        S = (A1 * A2) / D

        return 1 - S.mean()


'''
MSSSIML1Loss 구현 by gpt
sigma level만 수정하면 됨
'''

class MSSSIML1Loss(nn.Module):
    """
    MS-SSIM + L1 손실 모듈.
    """

    def __init__(self, win_size: int = 7, k1: float = 0.01, k2: float = 0.03, alpha: float = 0.84):
        """
        Args:
            win_size: SSIM 계산을 위한 윈도우 크기.
            k1: SSIM 계산을 위한 k1 파라미터.
            k2: SSIM 계산을 위한 k2 파라미터.
            alpha: 결합된 손실에서 MS-SSIM의 가중치.
        """
        super().__init__()
        self.win_size = win_size
        self.k1, self.k2 = k1, k2
        self.alpha = alpha
        self.register_buffer("w", torch.ones(1, 1, win_size, win_size) / win_size ** 2)
        NP = win_size ** 2
        self.cov_norm = NP / (NP - 1)
        self.sigma_levels = [1, 2, 4, 7]
        
        # 각 sigma 레벨에 대해 버퍼 등록
        for sigma in self.sigma_levels:
            self.register_buffer(f"w_{int(sigma)}", torch.ones(1, 1, int(sigma), int(sigma)) / sigma ** 2)
        
    def _calculate_ssim(self, X, Y, C1, C2, win_size):
        #print(f"Current w: {w}")
        w = getattr(self, f"w_{win_size}")
        print('\n w:',w)
        ux = F.conv2d(X, w, padding=win_size//2)
        uy = F.conv2d(Y, w, padding=win_size//2)
        uxx = F.conv2d(X * X, w, padding=win_size//2)
        uyy = F.conv2d(Y * Y, w, padding=win_size//2)
        uxy = F.conv2d(X * Y, w, padding=win_size//2)
        vx = (uxx - ux * ux)
        vy = (uyy - uy * uy)
        vxy = (uxy - ux * uy)
        A1, A2, B1, B2 = (
            2 * ux * uy + C1,
            2 * vxy + C2,
            ux ** 2 + uy ** 2 + C1,
            vx + vy + C2,
        )
        D = B1 * B2
        S = (A1 * A2) / D
        print("loss :",S.mean, "\nw :",w)
        return S.mean()

    def _calculate_msssim(self, X, Y, data_range):
        C1 = (self.k1 * data_range) ** 2
        C2 = (self.k2 * data_range) ** 2
        msssim = torch.ones_like(X)
        for sigma in self.sigma_levels:
            win_size = int(sigma)
            ssim = self._calculate_ssim(X, Y, C1, C2, win_size)
            msssim *= ssim
            
            # 디버깅을 위해 sigma 값과 w를 출력
            #print(f"Current sigma: {sigma}")
            #print(f"Current w: {w}")
        return msssim.mean()

    def forward(self, X, Y, data_range):
        X = X.unsqueeze(1)
        Y = Y.unsqueeze(1)
        data_range = data_range[:, None, None, None]

        msssim_loss = 1 - self._calculate_msssim(X, Y, data_range)

        # 가우시안 필터를 적용한 L1 손실 계산
        sigma_M = self.sigma_levels[-1]
        gaussian_filter = torch.ones(1, 1, int(sigma_M), int(sigma_M)) / sigma_M ** 2
        gaussian_filter = gaussian_filter.to(X.device)  # Move filter to the same device as X
        filtered_X = F.conv2d(X, gaussian_filter, padding=int(sigma_M//2))
        filtered_Y = F.conv2d(Y, gaussian_filter, padding=int(sigma_M//2))
        l1_loss = F.l1_loss(filtered_X, filtered_Y)

        loss = self.alpha * msssim_loss + (1 - self.alpha) * l1_loss

        return loss
