'''Python version of D2_ESN.'''

import torch
import numpy as np
import random
import cv2
from .functions import *
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = "cpu"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

seed = 11
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


class ESN_2D(torch.nn.Module):
    def __init__(self, input_dim=1, n_reservoir=100, spectral_radius=(0.9, 0.9), alpha=0.9,
                 connectivity=0.1, noise_level=1e-4, start_node=(1, 1), activation=torch.tanh):
        super(ESN_2D, self).__init__()
        self.n_reservoir = n_reservoir  # Number of reservoir neurons
        self.spectral_radius = spectral_radius  # Spectral radius for the reservoir
        self.alpha = alpha  # Ridge regression coefficient
        self.connectivity = connectivity
        self.noise_level = noise_level
        self.start_node = start_node
        self.activation = activation

        W_in = 2 * np.random.rand(input_dim, n_reservoir) - 1  # Input weights

        W_res_1 = np.random.rand(n_reservoir, n_reservoir) - 0.5  # Reservoir 内部权重
        mask = np.random.rand(n_reservoir, n_reservoir) < connectivity
        W_res_1 *= mask

        W_res_2 = np.random.rand(n_reservoir, n_reservoir) - 0.5  # Reservoir 内部权重
        mask = np.random.rand(n_reservoir, n_reservoir) < connectivity
        W_res_2 *= mask

        # Ensure the reservoir has the desired spectral radius
        rho_W_res = max(abs(np.linalg.eig(W_res_1)[0]))  # 谱半径最大特征值的绝对值
        W_res_1 *= self.spectral_radius[0] / rho_W_res

        rho_W_res = max(abs(np.linalg.eig(W_res_2)[0]))  # 谱半径最大特征值的绝对值
        W_res_2 *= self.spectral_radius[1] / rho_W_res

        # Convert weights to PyTorch tensors
        self.W_in = torch.nn.Parameter(torch.from_numpy(W_in).float(), requires_grad=False).to(device)
        self.W_res_1 = torch.nn.Parameter(torch.from_numpy(W_res_1).float(), requires_grad=False).to(device)
        self.W_res_2 = torch.nn.Parameter(torch.from_numpy(W_res_2).float(), requires_grad=False).to(device)

    # 更新水库状态
    def update_reservoir(self, x, state1, state2):
        """
        x: [batch_size]
        state: [batch_size, n_reservoir]
        """
        x = x.to(device)
        pre_activation = torch.mm(x.unsqueeze(1), self.W_in) + torch.mm(state1, self.W_res_1) + torch.mm(state2, self.W_res_2)
        state_new = self.activation(pre_activation)
        return state_new
    
    # 自定义岭回归
    def ridge_regression(self, states, targets):
        batch, length, dim = states.shape
        targets = targets.unsqueeze(2)  # Shape: [batch_size, length, 1]
        I = torch.eye(dim).unsqueeze(0).repeat(batch, 1, 1) * self.alpha
        I = I.to(states.device)

        XtX = torch.bmm(states.transpose(1, 2), states)
        XtX_plus_lambdaI = XtX + I
        XtX_plus_lambdaI_inv = torch.inverse(XtX_plus_lambdaI)
        Xty = torch.bmm(states.transpose(1, 2), targets)
        W_out = torch.bmm(XtX_plus_lambdaI_inv, Xty)

        return W_out

    def forward(self, input_image):
        """
        input_image: [batch_size, height, width]
        """
        input_image = input_image.detach().to(device) # 确保输入图像是一个tensor 且类型统一为torch.float32
        batch, height, width = input_image.shape
        state = torch.zeros((batch, height, width, self.n_reservoir), dtype=torch.float32).to(device)
        initial_state = torch.zeros((batch, self.n_reservoir), dtype=torch.float32).to(device)

        state[:, 0, 0, :] = self.update_reservoir(input_image[:, 0, 0], initial_state, initial_state)  # init
        # 先初始化第一行的所有state
        for t in range(1, width):
            state[:, 0, t, :] = self.update_reservoir(input_image[:, 0, t], state[:, 0, t-1, :], initial_state)
        # 每一列，一列一列地算
        for h in range(1, height):
            state[:, h, 0, :] = self.update_reservoir(input_image[:, h, 0], initial_state, state[:, h-1, 0, :])
            for t in range(1, width):
                state[:, h, t, :] = self.update_reservoir(input_image[:, h, t], state[:, h, t - 1, :], state[:, h-1, t, :])
        pre_states, targets = [], []
        for h in range(self.start_node[0], height):
            for t in range(self.start_node[1], width):
                new_state = torch.cat([state[:, h, t-1, :], state[:, h-1, t, :]], dim=1).unsqueeze(1) # 2D-ESN公式
                pre_states.append(new_state)
                targets.append(input_image[:, h, t].unsqueeze(1))

        pre_states = torch.cat(pre_states, dim=1)
        targets = torch.cat(targets, dim=1)

        return self.ridge_regression(pre_states, targets).squeeze(2)


def load_and_preprocess_image(file_path,):
    jpg_data = cv2.imread(file_path)[:, :, 0].astype(np.float32)  # 其实是取了图像的第0维
    height, width = jpg_data.shape
    jpg_data = cv2.resize(jpg_data, (height//4, width//4))  # 放缩成1/2的大小 注意要整除用“//”号
    return jpg_data


def normalize(image, normpr):
    """
    :param image: 原始图像数据，numpy数组格式
    :return: 归一化后的图像数据
    """
    # 将图像数据转换为float类型，以便进行除法操作
    image = image.astype(np.float32)
    # 图像归一化
    """这里不用除，放缩了图像会导致x过小（参考状态转移方程），进而ens输入学不到东西
    到底是第0维上取平均还是第1维上取平均还需进一步考虑"""
    std = np.std(image) * normpr
    normalized_image = (image - np.mean(image, axis=0)) / std   # (image - np.min(image))  / (np.max(image) - np.min(image))
    return normalized_image

