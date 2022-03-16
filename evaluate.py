import torch
import torch.nn as nn
import torch.autograd as A
import torch.optim as Optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import pandas as pd
import numpy as np

import os
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, roc_auc_score, confusion_matrix
import random
import sys

import argparse
import random


class Data(Dataset):
    def __init__(self, data_config):
        super(Data, self).__init__()
        self.root_path = data_config["root_path"]
        self.file_path = data_config["file_path"]
        with open(self.file_path, "r") as f:
            lines = f.readlines()
            self.folders = []
            for i in range(len(lines)):
                self.folders.append(lines[i].replace("\n", ""))

        # self.folders = os.listdir(self.root_path)
        self.pid_num = {}
        for i in range(len(self.folders)):
            self.pid_num[self.folders[i]] = i
        print("初始化完成...")

    def get_pid(self, pid):
        return self.pid_num[pid]

    def __len__(self):
        return len(self.folders)

    def __getitem__(self, item):
        pid = self.folders[item]
        files_path = os.path.join(self.root_path, pid)
        X = pd.read_csv(os.path.join(files_path, data_config["x_file"]), header=0, encoding="gbk")
        x = np.asarray(X.iloc[:, 1:]).reshape(-1, 116)

        M_X = pd.read_csv(os.path.join(files_path, "m0.csv"), header=0, encoding="gbk")
        m0 = np.asarray(M_X).reshape(-1, 116)

        Y = pd.read_csv(os.path.join(files_path, "y-tw1.csv"), header=0, encoding="gbk")
        Y = np.asarray(Y).reshape(-1, 112)
        # print(Y.shape)
        Time = pd.read_csv(os.path.join(files_path, "time.csv"), header=0, encoding="gbk")
        D = np.asarray(Time.iloc[:, 1]).reshape(-1, 1)
        T = np.asarray(Time.iloc[:, 0]).reshape(-1, 1)
        M_Y = pd.read_csv(os.path.join(files_path, "m1-tw1.csv"), header=0, encoding="gbk")
        m1 = np.asarray(M_Y).reshape(-1, 112)
        return x, Y, D, m1, T, m0


def collate_fn(Pack):
    Max_len = max([Pack[i][0].shape[0] for i in range(len(Pack))])
    XX, YY, DD, TT, MM1, MM0 = [], [], [], [], [], []
    for i in range(len(Pack)):
        # X 输入 Y 输出, D时间间隔 M1 Y的缺失指示 T时间  M0 X的缺失指示
        X, Y, D, M1, T, M0 = Pack[i] # 第几个病人的所有信息
        Y[np.isnan(Y)] = 0
        if not (X.shape[0] == Max_len):
            X = np.concatenate([X, np.tile(np.array([0]), (Max_len - X.shape[0], X.shape[1]))], axis=0)
            Y = np.concatenate([Y, np.tile(np.array([0]), (Max_len - Y.shape[0], Y.shape[1]))], axis=0)
            D = np.concatenate([D, np.tile(np.array([0]), (Max_len - D.shape[0], 1))], axis=0)
            T = np.concatenate([T, np.tile(np.array([0]), (Max_len - T.shape[0], 1))], axis=0)
            M1 = np.concatenate([M1, np.tile(np.array([0]), (Max_len - M1.shape[0], M1.shape[1]))], axis=0)
            M0 = np.concatenate([M0, np.tile(np.array([0]), (Max_len - M0.shape[0], M0.shape[1]))], axis=0)

        XX.append(X)
        YY.append(Y)
        DD.append(D)
        TT.append(T)
        MM1.append(M1)
        MM0.append(M0)

    XX = torch.from_numpy(np.stack(XX, axis=0)).to(torch.float32)
    YY = torch.from_numpy(np.stack(YY, axis=0)).to(torch.int64)
    DD = torch.from_numpy(np.stack(DD, axis=0)).to(torch.float32)
    TT = torch.from_numpy(np.stack(TT, axis=0)).to(torch.float32)
    MM0 = torch.from_numpy(np.stack(MM0, axis=0)).to(torch.float32)
    try:
        MM1 = torch.from_numpy(np.stack(MM1, axis=0)).to(torch.float32)
    except:
        print([MM1[i].shape for i in range(len(MM1))])
        sys.exit()
    return XX, YY, DD, TT, MM1, MM0


class Model2(nn.Module):
    def __init__(self, model_config):

        super(Model2, self).__init__()

        self.input_size = model_config['input_size']
        self.hidden_size = model_config['hidden_size']
        self.num_class = model_config['num_class']

        self.gru1 = nn.GRUCell(input_size=self.input_size, hidden_size=self.hidden_size)
        self.gru2 = nn.GRUCell(input_size=self.hidden_size, hidden_size=self.hidden_size)
        self.gru3 = nn.GRUCell(input_size=self.hidden_size, hidden_size=self.hidden_size)

        # self.BN_vet_1 = nn.BatchNorm1d(self.hidden_size)  # 纵向的 1-2 层 batch norm 1d
        # self.BN_vet_2 = nn.BatchNorm1d(self.hidden_size)  # 纵向的 2-3 层 batch norm 1d
        # self.BN_hor = nn.BatchNorm1d(self.hidden_size)  # 横向的 batch_norm 1d

        self.output = nn.Sequential(  # 输出 y
            nn.Linear(in_features=self.hidden_size, out_features=self.num_class),
            nn.Softmax(dim=1)
        )

        self.output_x = nn.Sequential(  # 输出 x
            nn.Linear(in_features=self.hidden_size, out_features=self.input_size)
        )
        
        self.output_m = nn.Sequential( # 输出m
            nn.Linear(in_features=self.hidden_size, out_features=self.input_size),
            nn.Sigmoid() # batch,time-1,var
        )

        self.decay = nn.Sequential(  # 时间衰减函数
            nn.Linear(in_features=1, out_features=self.hidden_size),
            nn.ReLU()
        )

        self.init_weights()  # 初始化网络权重

    def forward(self, inputs):
        # x 的第一个时间点不能有缺失, m==0表示缺失
        x, d, m1 = inputs  # 接受输入

        device = x.device  # x 的设备 cpu / gpu
        Batch_size, time_len, var_len = x.shape  # 各维度的长度
        h0_1 = torch.tensor(np.random.normal(0, 0.1, (Batch_size, self.hidden_size)), dtype=torch.float32).to(
            device)  # 第一层GRU的h0
        h0_2 = torch.tensor(np.random.normal(0, 0.1, (Batch_size, self.hidden_size)), dtype=torch.float32).to(
            device).to(
            device)  # 第二层GRU的h0
        h0_3 = torch.tensor(np.random.normal(0, 0.1, (Batch_size, self.hidden_size)), dtype=torch.float32).to(
            device).to(
            device)  # 第三层GRU的h0

        Y = []  # 每个时间点的输出结果
        X_hat = []  # 每个时间点估计的 X 的输出，第一个时间点就是 X 本身
        M_hat = [] # 每个时间点估计的 M 的输出
        for i1 in range(time_len):  # 沿着时间维度

            if i1 > 0:
                gamma = torch.exp(-self.decay(d[:, i1, :]))  # (batch, 1) --> (batch, hidden)
                h0_1 = h0_1 * gamma # 时间衰减
                h0_2 = h0_2 * gamma # 时间衰减
                h0_3 = h0_3 * gamma # 时间衰减
            if i1 == 0: # 如果是第一次，就不需要缺失值替换
                x1 = x[:, i1, :]  # (batch, var)
                # X_hat.append(x1)
            elif i1 > 0: # 如果是第二次，需要进行缺失值替换
                x1 = x_hat * (1-m1[:, i1, :]) + m1[:, i1, :] * x[:, i1, :]  # 缺失值替换
            
            ind = torch.where(torch.isnan(h0_1)) # 看看h0_1有没有缺失
            if len(ind[0])>0:
                print("初始隐藏层1  的缺失",ind)
                sys.exit()
            
            ind = torch.where(torch.isnan(x1)) # 看看h0_1有没有缺失
            if len(ind[0])>0:
                print("输入x1 的缺失",ind)
                sys.exit()
            
            h1 = self.gru1(x1, h0_1)  # 第一层 GRU
            ind = torch.where(torch.isnan(h1)) # 看看h1有没有缺失
            if len(ind[0])>0:
                print("第一层gru输出 的缺失",ind)
                sys.exit()
            
            
            h0_1 = h1  # 更新第一层GRU 的 h0
            # h0_1 = self.BN_hor(h0_1) # 横向 batchnorm 1d 更新h0
            # h1 = self.BN_vet_1(h1)
            
            ind = torch.where(torch.isnan(h1)) # 看看h1有没有缺失
            if len(ind[0])>0:
                print("第一层batchnorm 缺失 的缺失",ind)
                sys.exit()
                
            h1 = self.gru2(h1, h0_2)  # 第二层 GRU
            h0_2 = h1  # 更新第二层GRU 的 h0
            # h0_2 = self.BN_hor(h0_2) # 横向 batchnorm 1d 更新h0
            
            ind = torch.where(torch.isnan(h0_2)) # 看看h0_2有没有缺失
            if len(ind[0])>0:
                print("隐藏层 2 的缺失",ind)
                sys.exit()
            # h1 = self.BN_vet_2(h1)
            h1 = self.gru3(h1, h0_3)  # 第三层 GRU
            h0_3 = h1  # 更新第三层GRU 的 h0
            # h0_3 = self.BN_hor(h0_3) # 横向 batchnorm 1d 更新h0
            
            ind = torch.where(torch.isnan(h0_3)) # 看看h0_3有没有缺失
            if len(ind[0])>0:
                print("隐藏层  3 的缺失",ind)
                sys.exit()
            
            ind = torch.where(torch.isnan(h1)) # 看看h1有没有缺失
            if len(ind[0])>0:
                print("h1的缺失",ind)
                sys.exit()
            
            y = self.output(h1)  # 输出y
            # ind = torch.where(torch.isnan(y))# 看看y有没有缺失
            # if len(ind[0])>0:
                # print("y的缺失",ind)
            
            x_hat = self.output_x(h1)  # 输出 x 的估计值
            m_hat = self.output_m(h1)  # 输出m 的估计值
            
            Y.append(y)  # 保留 y

            if i1 >= 1:
                X_hat.append(x_hat) # 保留 xhat
                M_hat.append(m_hat) # 保留 mhat (batch,time-1,var)

        return torch.stack(Y, dim=1), torch.stack(X_hat, dim=1), torch.stack(M_hat, dim=1) # output - y, x, m

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear)):
                nn.init.xavier_uniform_(m.weight.data)
                nn.init.uniform_(m.bias.data, 0, 1)
            elif isinstance(m, (nn.GRUCell)):
                for name, param in m.named_parameters():
                    if name.find("weight") > 0:
                        nn.init.orthogonal(param.data)
                    else:
                        nn.init.zeros_(param.data)
            elif isinstance(m, (nn.BatchNorm1d)):
                nn.init.ones_(m.weight.data)
                nn.init.zeros_(m.bias.data)
            else:
                for param in m.parameters():
                    nn.init.normal_(param.data)


class Model0(nn.Module):
    def __init__(self, model_config):

        super(Model0, self).__init__()

        self.input_size = model_config['input_size']
        self.hidden_size = model_config['hidden_size']
        self.num_class = model_config['num_class']

        
        self.gru1 = nn.GRUCell(input_size=self.input_size, hidden_size=self.hidden_size)
        self.gru2 = nn.GRUCell(input_size=self.hidden_size, hidden_size=self.hidden_size)
        self.gru3 = nn.GRUCell(input_size=self.hidden_size, hidden_size=self.hidden_size)
        
        self.output = nn.Sequential(  # 输出 y
            nn.Linear(in_features=self.hidden_size, out_features=self.num_class),
                         
            nn.Softmax(dim=1)
        )
        self.decay = nn.Sequential(  # 时间衰减函数
            nn.Linear(in_features=1, out_features=self.hidden_size),
            nn.ReLU()
        )

        self.init_weights()

    def forward(self, inputs):
        x, d = inputs
        device = x.device
        Batch_size, time_len, var_len = x.shape                      
        device = x.device  # x 的设备 cpu / gpu
        Batch_size, time_len, var_len = x.shape  # 各维度的长度
        h0_1 = torch.tensor(np.random.normal(0, 0.1, (Batch_size, self.hidden_size)), dtype=torch.float32).to(
            device)  # 第一层GRU的h0
        h0_2 = torch.tensor(np.random.normal(0, 0.1, (Batch_size, self.hidden_size)), dtype=torch.float32).to(
            device).to(
            device)  # 第二层GRU的h0
        h0_3 = torch.tensor(np.random.normal(0, 0.1, (Batch_size, self.hidden_size)), dtype=torch.float32).to(
            device).to(
            device)  # 第三层GRU的h0
        
        
        Y = []  # 每个时间点的输出结果
        for i1 in range(time_len):  # 沿着时间维度

            if i1 > 0:
                gamma = torch.exp(-self.decay(d[:, i1, :]))  # (batch, 1) --> (batch, hidden)
                h0_1 = h0_1 * gamma # 时间衰减
                h0_2 = h0_2 * gamma # 时间衰减
                h0_3 = h0_3 * gamma # 时间衰减
            
            x1 = x[:, i1, :]
            
            
            h1 = self.gru1(x1, h0_1)  # 第一层 GRU            
            h0_1 = h1  # 更新第一层GRU 的 h0

            h1 = self.gru2(h1, h0_2)  # 第二层 GRU
            h0_2 = h1  # 更新第二层GRU 的 h0

            h1 = self.gru3(h1, h0_3)  # 第三层 GRU
            h0_3 = h1  # 更新第三层GRU 的 h0

            
            y = self.output(h1)  # 输出y

            
            Y.append(y)  # 保留 y

        return torch.stack(Y, dim=1)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear)):
                nn.init.xavier_uniform_(m.weight.data)
                nn.init.uniform_(m.bias.data, 0, 1)
            elif isinstance(m, (nn.GRUCell)):
                for name, param in m.named_parameters():
                    if name.find("weight") > 0:
                        nn.init.orthogonal(param.data)
                    else:
                        nn.init.zeros_(param.data)
            elif isinstance(m, (nn.BatchNorm1d)):
                nn.init.ones_(m.weight.data)
                nn.init.zeros_(m.bias.data)
            else:
                for param in m.parameters():
                    nn.init.normal_(param.data)


class Model2_3(nn.Module):
    def __init__(self, model_config):

        super(Model2_3, self).__init__()

        self.input_size = model_config['input_size']
        self.hidden_size = model_config['hidden_size']
        self.num_class = model_config['num_class']

        self.gru1 = nn.GRUCell(input_size=self.input_size, hidden_size=self.hidden_size)
        self.gru2 = nn.GRUCell(input_size=self.hidden_size, hidden_size=self.hidden_size)
        self.gru3 = nn.GRUCell(input_size=self.hidden_size, hidden_size=self.hidden_size)

        # self.BN_vet_1 = nn.BatchNorm1d(self.hidden_size)  # 纵向的 1-2 层 batch norm 1d
        # self.BN_vet_2 = nn.BatchNorm1d(self.hidden_size)  # 纵向的 2-3 层 batch norm 1d
        # self.BN_hor = nn.BatchNorm1d(self.hidden_size)  # 横向的 batch_norm 1d

        self.output = nn.Sequential(  # 输出 y
            nn.Linear(in_features=self.hidden_size, out_features=self.num_class),
            nn.Softmax(dim=1)
        )

        self.output_x = nn.Sequential(  # 输出 x
            nn.Linear(in_features=self.hidden_size, out_features=self.input_size)
        )
        
        self.output_m = nn.Sequential( # 输出m
            nn.Linear(in_features=self.hidden_size, out_features=self.input_size),
            nn.Sigmoid() # batch,time-1,var
        )

        self.decay = nn.Sequential(  # 时间衰减函数
            nn.Linear(in_features=1, out_features=self.hidden_size),
            nn.ReLU()
        )

        self.init_weights()  # 初始化网络权重

    def forward(self, inputs):
        # x 的第一个时间点不能有缺失, m==0表示缺失
        x, d, m1 = inputs  # 接受输入

        device = x.device  # x 的设备 cpu / gpu
        Batch_size, time_len, var_len = x.shape  # 各维度的长度
        h0_1 = torch.tensor(np.random.normal(0, 0.1, (Batch_size, self.hidden_size)), dtype=torch.float32).to(
            device)  # 第一层GRU的h0
        h0_2 = torch.tensor(np.random.normal(0, 0.1, (Batch_size, self.hidden_size)), dtype=torch.float32).to(
            device).to(
            device)  # 第二层GRU的h0
        h0_3 = torch.tensor(np.random.normal(0, 0.1, (Batch_size, self.hidden_size)), dtype=torch.float32).to(
            device).to(
            device)  # 第三层GRU的h0

        Y = []  # 每个时间点的输出结果
        X_hat = []  # 每个时间点估计的 X 的输出，第一个时间点就是 X 本身
        M_hat = [] # 每个时间点估计的 M 的输出
        for i1 in range(time_len):  # 沿着时间维度

            if i1 > 0:
                gamma = torch.exp(-self.decay(d[:, i1, :]))  # (batch, 1) --> (batch, hidden)
                h0_1 = h0_1 * gamma # 时间衰减
                h0_2 = h0_2 * gamma # 时间衰减
                h0_3 = h0_3 * gamma # 时间衰减
            if i1 == 0: # 如果是第一次，就不需要缺失值替换
                x1 = x[:, i1, :]  # (batch, var)
                # X_hat.append(x1)
            elif i1 > 0: # 如果是第二次，需要进行缺失值替换
                x1 = x_hat * (1-m1[:, i1, :]) + m1[:, i1, :] * x[:, i1, :]  # 缺失值替换
            
            ind = torch.where(torch.isnan(h0_1)) # 看看h0_1有没有缺失
            if len(ind[0])>0:
                print("初始隐藏层1  的缺失",ind)
                sys.exit()
            
            ind = torch.where(torch.isnan(x1)) # 看看h0_1有没有缺失
            if len(ind[0])>0:
                print("输入x1 的缺失",ind)
                sys.exit()
            
            h1 = self.gru1(x1, h0_1)  # 第一层 GRU
            ind = torch.where(torch.isnan(h1)) # 看看h1有没有缺失
            if len(ind[0])>0:
                print("第一层gru输出 的缺失",ind)
                sys.exit()
            
            
            h0_1 = h1  # 更新第一层GRU 的 h0
            # h0_1 = self.BN_hor(h0_1) # 横向 batchnorm 1d 更新h0
            # h1 = self.BN_vet_1(h1)
            
            ind = torch.where(torch.isnan(h1)) # 看看h1有没有缺失
            if len(ind[0])>0:
                print("第一层batchnorm 缺失 的缺失",ind)
                sys.exit()
                
            h1 = self.gru2(h1, h0_2)  # 第二层 GRU
            h0_2 = h1  # 更新第二层GRU 的 h0
            # h0_2 = self.BN_hor(h0_2) # 横向 batchnorm 1d 更新h0
            
            ind = torch.where(torch.isnan(h0_2)) # 看看h0_2有没有缺失
            if len(ind[0])>0:
                print("隐藏层 2 的缺失",ind)
                sys.exit()
            # h1 = self.BN_vet_2(h1)
            h1 = self.gru3(h1, h0_3)  # 第三层 GRU
            h0_3 = h1  # 更新第三层GRU 的 h0
            # h0_3 = self.BN_hor(h0_3) # 横向 batchnorm 1d 更新h0
            
            ind = torch.where(torch.isnan(h0_3)) # 看看h0_3有没有缺失
            if len(ind[0])>0:
                print("隐藏层  3 的缺失",ind)
                sys.exit()
            
            ind = torch.where(torch.isnan(h1)) # 看看h1有没有缺失
            if len(ind[0])>0:
                print("h1的缺失",ind)
                sys.exit()
            
            y = self.output(h1)  # 输出y
            # ind = torch.where(torch.isnan(y))# 看看y有没有缺失
            # if len(ind[0])>0:
                # print("y的缺失",ind)
            
            x_hat = self.output_x(h1)  # 输出 x 的估计值
            m_hat = self.output_m(h1)  # 输出m 的估计值
            
            Y.append(y)  # 保留 y

            if i1 >= 1:
                X_hat.append(x_hat) # 保留 xhat
                M_hat.append(m_hat) # 保留 mhat (batch,time-1,var)

        return torch.stack(Y, dim=1), torch.stack(X_hat, dim=1), torch.stack(M_hat, dim=1) # output - y, x, m

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear)):
                nn.init.xavier_uniform_(m.weight.data)
                nn.init.uniform_(m.bias.data, 0, 1)
            elif isinstance(m, (nn.GRUCell)):
                for name, param in m.named_parameters():
                    if name.find("weight") > 0:
                        nn.init.orthogonal(param.data)
                    else:
                        nn.init.zeros_(param.data)
            elif isinstance(m, (nn.BatchNorm1d)):
                nn.init.ones_(m.weight.data)
                nn.init.zeros_(m.bias.data)
            else:
                for param in m.parameters():
                    nn.init.normal_(param.data)



class Model5(nn.Module):
    def __init__(self, model_config):

        super(Model5, self).__init__()

        self.input_size = model_config['input_size']
        self.hidden_size = model_config['hidden_size']
        self.num_class = model_config['num_class']

        self.gru1 = nn.GRUCell(input_size=self.input_size, hidden_size=self.hidden_size)
        self.gru2 = nn.GRUCell(input_size=self.hidden_size, hidden_size=self.hidden_size)
        self.gru3 = nn.GRUCell(input_size=self.hidden_size, hidden_size=self.hidden_size)


        self.output = nn.Sequential(  # 输出 y
            nn.Linear(in_features=self.hidden_size, out_features=self.num_class),
            nn.Softmax(dim=1)
        )

        self.decay_x = nn.Sequential(  # 时间衰减函数 x 
            nn.Linear(in_features=1, out_features=self.input_size),
            nn.ReLU()
        )
        
        self.decay_h = nn.Sequential(  # 时间衰减函数 h
            nn.Linear(in_features=1, out_features=self.hidden_size),
            nn.ReLU()
        )

        self.init_weights()  # 初始化网络权重

    def forward(self, inputs):
        # x 的第一个时间点不能有缺失, m==0表示缺失
        x, d, m1 = inputs  # 接受输入

        device = x.device  # x 的设备 cpu / gpu
        Batch_size, time_len, var_len = x.shape  # 各维度的长度
        h0_1 = torch.tensor(np.random.normal(0, 0.1, (Batch_size, self.hidden_size)), dtype=torch.float32).to(
            device)  # 第一层GRU的h0
        h0_2 = torch.tensor(np.random.normal(0, 0.1, (Batch_size, self.hidden_size)), dtype=torch.float32).to(
            device).to(
            device)  # 第二层GRU的h0
        h0_3 = torch.tensor(np.random.normal(0, 0.1, (Batch_size, self.hidden_size)), dtype=torch.float32).to(
            device).to(
            device)  # 第三层GRU的h0

        Y = []  # 每个时间点的输出结果
        X_hat = []  # 每个时间点估计的 X 的输出，第一个时间点就是 X 本身
        M_hat = [] # 每个时间点估计的 M 的输出
        for i1 in range(time_len):  # 沿着时间维度

            if i1 > 0:
                gamma_h = torch.exp(-self.decay_h(d[:, i1, :]))  # (batch, 1) --> (batch, hidden)
                h0_1 = h0_1 * gamma_h # 时间衰减
                h0_2 = h0_2 * gamma_h # 时间衰减
                h0_3 = h0_3 * gamma_h # 时间衰减
            if i1 == 0: # 如果是第一次，就不需要缺失值替换
                x1 = x[:, i1, :]  # (batch, var)
                # X_hat.append(x1)
            elif i1 > 0: # 如果是第二次，需要进行缺失值替换
                gamma_x = torch.exp(-self.decay_x(d[:, i1, :]))  # (batch, 1) --> (batch, input_size)
                x1 = (x[:,i1-1,:] * gamma_x + (1-gamma_x) * x[:,0,:]) * (1-m1[:, i1, :]) + m1[:, i1, :] * x[:, i1, :]  # 缺失值替换
            
            h1 = self.gru1(x1, h0_1)  # 第一层 GRU
          
            h0_1 = h1  # 更新第一层GRU 的 h0
            
            h1 = self.gru2(h1, h0_2)  # 第二层 GRU
            
            h0_2 = h1  # 更新第二层GRU 的 h0
          
            h1 = self.gru3(h1, h0_3)  # 第三层 GRU
            
            h0_3 = h1  # 更新第三层GRU 的 h0
            
            y = self.output(h1)  # 输出y
            
            Y.append(y)  # 保留 y


        return torch.stack(Y, dim=1) # output - y

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear)):
                nn.init.xavier_uniform_(m.weight.data)
                nn.init.uniform_(m.bias.data, 0, 1)
            elif isinstance(m, (nn.GRUCell)):
                for name, param in m.named_parameters():
                    if name.find("weight") > 0:
                        nn.init.orthogonal(param.data)
                    else:
                        nn.init.zeros_(param.data)
            elif isinstance(m, (nn.BatchNorm1d)):
                nn.init.ones_(m.weight.data)
                nn.init.zeros_(m.bias.data)
            else:
                for param in m.parameters():
                    nn.init.normal_(param.data)




def get_argment(): # 测试的参数
    parser = argparse.ArgumentParser()
    # 模型参数地址
    parser.add_argument("--load", help="load the model parameters", type=str, required=True)
    # 训练次数
    parser.add_argument("--epoch", help="epochs", type=int)
    # 第几例y变量
    parser.add_argument("--col-y", help="column index of y to model", type=int)
    # 类别不平衡权重
    parser.add_argument("--weight", help="weight for focal loss", nargs="+", type=float)
    # batch size
    parser.add_argument("--batch-size", help="batch size", type=int)
    # 学习率
    parser.add_argument("--lr", help="learning rate", type=float, default=0.01)
    # 学习率的衰减 比例
    parser.add_argument("--gamma-lr", help="gamma for lr to decay", type=float, default=0.5)
    # focal loss 的聚焦因子
    parser.add_argument("--gamma-fl", help="gamma for focal loss", type=float, default=2.0)
    # 多少次循环 来衰减 一次 学习率 || 若不想改变学习率 则可以将 n_step 设置的很大 例如 100000000
    parser.add_argument("--n-step", help="steps to decay the lr", type=int)
    # 从第几个循环开始训练
    parser.add_argument("--start-epoch", help="epoch number to start", type=int, default=0)
    # 根目录 主要用于输出 参数
    parser.add_argument("--root", help="root path", type=str, default="/home/luojiawei/multimodal_model/")
    # 二范数正则化的  系数
    parser.add_argument("--lambda-l2", help="lambda for L2 norm", type=float, default=0.001)
    # xhat loss 的lambda-x
    parser.add_argument("--lambda-x", help="lambda for xhat loss", type=float, default=0.001)
    # mhat loss 的lambda-m
    parser.add_argument("--lambda-m", help="lambda for mhat loss", type=float, default=0.001)
    # 选择要测试的模型
    parser.add_argument("--model", help="select the model to test", type=str, required=True)
    # 备注信息
    parser.add_argument("--note", help="Additional comments required", type=str, default=None)
    # 输出的 x 的类型
    parser.add_argument("--x-file", help="type of input x", type=str, default="x.csv")
    
    return parser.parse_args()


def torch_normalize(x):
    # torch tensor 数据归一化
    x1 = x.numpy()
    if len(x.shape) == 3:
        ma = np.nanmax(x1, axis=(0, 1), keepdims=True)
        mi = np.nanmin(x1, axis=(0, 1), keepdims=True)
        x2 = (x1 - mi) / (ma - mi + 1e-10)
        x2 = np.clip(x2, 0, 1)
    elif len(x.shape) == 2:
        ma = np.nanmax(x1, axis=0, keepdims=True)
        mi = np.nanmin(x1, axis=0, keepdims=True)
        x2 = (x1 - mi) / (ma - mi + 1e-10)
        x2 = np.clip(x2, 0, 1)
    return torch.from_numpy(x2)


if __name__ == "__main__":
    
    
    models = {
        "model2":Model2,
        "model0":Model0,
        "model2_3":Model2_3,
        "model5":Model5,
    }
    
    args = get_argment()  # 获取训练参数
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("using {}".format(device.type))
    
    
    data_config = { # 测试集数据
        "root_path": "/home/luojiawei/multimodal_model/data/新的数据清洗 20211204/clean_data/test",
        "file_path" : "/home/luojiawei/multimodal_model/data/新的数据清洗 20211204/测试集ID.txt",
        "x_file" : args.x_file,
        # "root_path": "{}/data/新的数据清洗 20211204/clean_data/test".format(args.root),
    }
    data_te = Data(data_config) # 测试集数据
    dataloader_te = DataLoader(data_te, batch_size=args.batch_size, collate_fn=collate_fn, shuffle=True)
    
    model_config = {
        # 模型配置
        "input_size": 116, # 输出变量维度
        "hidden_size": 400, # 隐藏层维度
        "num_class": 2, # 输出维度
    }

    model = models[args.model](model_config).to(device) # 模型初始化

    # 因为是测试阶段，所有必须要预先加载
    print("load the parameters from {}".format(args.load))
    model.load_state_dict(torch.load(args.load))
        
    col = int(args.col_y) # 第几列 y 变量
    
    print("开始测试模型")
        
    model.eval() # 测试模式
    with torch.no_grad(): # 不计算梯度
    
        
        Y_true, Y_pred = [], [] # (batch * time, 1)
        Time = []
        for i, (x, y, d, t, m1, m0) in enumerate(dataloader_te):
            # t (batch, time, 1)
            x = torch_normalize(x) # 归一化 0-1
            
            y = y.to(device) # batch,time, ncol
            m1 = m1.to(device)
            m0 = m0.to(device) # batch, time, var

            # 运行前向过程  根据不同的模型运行不同的函数
            if args.model in ["model2", "model2_3"]:
                yhat, _, _ = model([x.to(device), d.to(device), m0])
            elif args.model == "model0":
                yhat = model([x.to(device), d.to(device)])
            elif args.model == "model5":
                yhat = model([x.to(device), d.to(device), m0])
            else:
                print("该模型不支持测试")
                sys.exit()
            
            Time.append(t[m1[...,col] == 1].data) # 添加 t
            Y_pred.append(yhat[m1[..., col] == 1].cpu().data) # 添加y的预测值
            Y_true.append(y[..., col][m1[..., col] == 1].cpu().data) # 添加y的真值
            print(i)
            
           
    Y_pred = torch.cat(Y_pred, dim=0).numpy().reshape(-1,2) # 转成numpy数组 batch*time, 2
    Y_true = torch.cat(Y_true, dim=0).numpy().reshape(-1,1) # batch*time, 1
    Time = torch.cat(Time, dim=0).numpy().reshape(-1,1) # batch*time, 1
    
    # 合并预测和真值
    Y_comb = np.concatenate([Time, Y_pred, Y_true], axis=1) # batch*time, 3
    Y_comb1 = pd.DataFrame(Y_comb, columns=["time","pred","pred1","true"])
    
    # 权重的文件名
    r = os.path.split(args.load)[1]
    
    if args.note:  # 如果有备注信息， 则将备注信息加在文件名最后
        Y_comb1.to_excel(os.path.join(args.root,"new_model2",
                                        "results",
                                        r.replace("param-","").replace(".pth","")+"结果-"+args.note+".xlsx"),
                    index=False)
        print("文件{}已经输出".format(r.replace("param-","").replace(".pth","")+"结果-"+args.note+".xlsx"))
    else:
        Y_comb1.to_excel(os.path.join(args.root,"new_model2",
                                        "results",
                                        r.replace("param-","").replace(".pth","")+"结果.xlsx"),
                    index=False)
        print("文件{}已经输出".format(r.replace("param-","").replace(".pth","")+"结果.xlsx"))














    
    
    
    
    
    