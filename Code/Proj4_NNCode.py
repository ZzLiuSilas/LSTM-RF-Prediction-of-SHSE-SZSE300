# -*- coding: utf-8 -*-
"""
Created on Tue May 24 14:03:45 2022

@author: Silas Liew
"""

import torch
import torch.nn as nn
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from torch.nn.utils.rnn import pad_sequence
#定义超参
torch.manual_seed(0)
np.random.seed(0)

input_window = 30
output_window = 1
batch_size = 30
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
#构建LSTM模型
class LSTM(nn.Module):
    def __init__(self, input_dim = 30, hidden_layer_dim = 100, output_dim = 1):
        super().__init__()
        
        self.hidden_layer_dim = hidden_layer_dim
        
        self.lstm = nn.LSTM(input_dim, hidden_layer_dim).cuda()
        #添加一线性层用于从隐藏层输出
        self.linear = nn.Linear(hidden_layer_dim, output_dim).cuda()
        
        self.hidden_cell = self.init_hidden()
    
    def init_hidden(self):    
        return(torch.zeros(1, 1, self.hidden_layer_dim).cuda(),
               torch.zeros(1, 1, self.hidden_layer_dim).cuda())
    
    def forward(self, input_seq):
        lstm_out, self.hidden_cell = self.lstm(input_seq.view(len(input_seq), 1, -1), 
                                               self.hidden_cell)
        
        predictions = self.linear(lstm_out.view(len(input_seq), -1))
        
        return predictions[-1]
#数据预处理
##窗口划分
def create_inout_sequences(input_data, tw):
    inout_seq = []
    L = len(input_data)
    for i in range(L - tw):
        train_seq = input_data[i:i + tw]
        train_label = input_data[i + output_window:i + tw + output_window]
        inout_seq.append((train_seq, train_label))
    return torch.FloatTensor(inout_seq)
##读入、划分数据
def get_data():
    series = pd.read_excel('./Proj4_data.xlsx')
    date = pd.DatetimeIndex(series["date"])
    series = pd.Series(np.array(series["close"]), index = date)
    #归一标准化
    scaler = MinMaxScaler(feature_range=(-1, 1))
    series = scaler.fit_transform(series.values.reshape(-1, 1)).reshape(-1)
    #划分训练集、测试集
    train_samples = int(0.8 * len(series))
    train_data = series[:train_samples]
    test_data = series[train_samples:]
    #转换为张量
    train_sequence = create_inout_sequences(train_data, input_window)
    train_sequence = train_sequence[:-output_window]

    test_data = create_inout_sequences(test_data, input_window)
    test_data = test_data[:-output_window]

    return train_sequence.to(device), test_data.to(device)
##划定批次、标签
def get_batch(source, i, batch_size):
    seq_len = min(batch_size, len(source) - 1 - i)
    data = source[i:i + seq_len]
    input = torch.stack(torch.stack([item[0] for item in data]).chunk(input_window, 1))
    target = torch.stack(torch.stack([item[1] for item in data]).chunk(input_window, 1))
    return input, target

#训练
def train(train_data):
    model.train()   
    for batch_index, i in enumerate(range(0, len(train_data) - 1, batch_size)):
    
        #按批获取数据、标签
        batch_data, targets = get_batch(train_data, i, batch_size)
        #总损失
        total_loss = 0 
        #隐藏层、梯度清零
        model.hidden_cell = model.init_hidden()        
        optimizer.zero_grad()
        #前馈并计算损失
        output = model(batch_data)
        loss = criterion(output, targets)
        #反向传播
        loss.backward()
        #梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.7)
        
        optimizer.step()
        #打印训练步骤
        total_loss += loss.item()
        log_interval = int(len(train_data) / batch_size / 5)
        if batch_index % log_interval == 0 and batch_index > 0:
            cur_loss = total_loss / log_interval
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.6f} | loss {:5.5f} | ppl {:8.2f}'
                  .format(epoch, batch_index, len(train_data) // batch_size, scheduler.get_lr()[0], cur_loss, math.exp(cur_loss)))
            
        

#评估
def evaluate(eval_model, data_source):
    eval_model.eval()
    total_loss = 0
    eval_batch_size = 30
    with torch.no_grad():
        for i in range(0, len(data_source) - 1, eval_batch_size):
            data, targets = get_batch(data_source, i, eval_batch_size)
            data = pad_sequence(data)
            output = eval_model(data)
            total_loss += len(data[0]) * criterion(output, targets).cpu().item()
    return total_loss / len(data)
#作图、保存数据
def plot_and_loss(eval_model, data_source, epoch):
    eval_model.eval()
    total_loss = 0.
    test_result = torch.Tensor(0)
    truth = torch.Tensor(0)
    with torch.no_grad():
        for i in range(0, len(data_source) - 1):
            data, target = get_batch(data_source, i, 1)
            data = pad_sequence(data)
            output = eval_model(data)
            total_loss += criterion(output, target).item()
            test_result = torch.cat((test_result, output[-1].view(-1).cpu()), 0)
            truth = torch.cat((truth, target[-1].view(-1).cpu()), 0)
            
    fig_size = plt.rcParams["figure.figsize"]
    fig_size[0] = 15
    fig_size[1] = 7.5
    plt.plot(test_result, color="red")
    plt.plot(truth, color="blue")
    plt.grid(True, which="both")
    plt.axhline(y=0, color="k")
    plt.savefig('LSTM-epoch%d.png' % epoch)
    plt.show()
    plt.close()
    
    #pd.DataFrame(test_result.tolist()).to_csv("Proj4_Result_LSTM.csv", index = True , sep = ",")
    #pd.DataFrame(truth.tolist()).to_csv("Proj4_Truth_LSTM.csv", index = True , sep = ",")
    
    return total_loss / i
#定义训练参数
train_data, val_data = get_data()
model = LSTM().to(device)
print(model)
criterion = nn.MSELoss().to(device)
lr = 0.02
gamma = 0.96
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma = gamma)
epochs = 150
#训练循环
for epoch in range(1, epochs + 1):
    
    train(train_data)
    
    if(epoch == epochs):
        val_loss = plot_and_loss(model, val_data, epoch)
    else:
        val_loss = evaluate(model, val_data)
        
    print('-' * 65)
    print('| end of epoch {:3d} | valid loss {:5.5f} | valid ppl {:8.2f}'.format(epoch, val_loss, math.exp(val_loss)))
    print('-' * 65)
    
    scheduler.step()
