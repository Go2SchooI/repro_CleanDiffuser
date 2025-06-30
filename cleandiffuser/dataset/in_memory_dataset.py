import torch
from torch.utils.data import Dataset
import h5py
import numpy as np

class InMemoryH5Dataset(Dataset):
    def __init__(self, hdf5_path):
        # 打开 hdf5 文件并加载所有内容到内存
        with h5py.File(hdf5_path, 'r') as f:
            # 图像信息，这里假设你只使用 'agentview' 相机图像（形如 N x H x W x C）
            self.imgs = f['agentview'][:]  # shape: (N, 84, 84, 3)
            self.states = f['robot0_states'][:]    # shape: (N, state_dim)
            self.actions = f['robot0_actions'][:]  # shape: (N, action_dim)

        self.len = self.imgs.shape[0]

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        # 转成 Tensor，图像转为 CHW
        img = torch.from_numpy(self.imgs[idx]).permute(2, 0, 1).float() / 255.0  # normalize to [0, 1]
        state = torch.from_numpy(self.states[idx]).float()
        action = torch.from_numpy(self.actions[idx]).float()
        return {"image": img, "state": state, "action": action}
