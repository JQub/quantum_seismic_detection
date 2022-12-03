from distutils.command.build_scripts import first_line_re
import torch
import torch.nn.functional as F
import torch.optim as optim
import argparse

import pytorch_quantum.torchquantum as tq
import pytorch_quantum.torchquantum.functional as tqf
import random
from pytorch_quantum.examples.core.datasets import MNIST
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset,DataLoader
import sys
import numpy as np
from wave_qmodel import QFCModel,QRNN,QRNNBlockVqc,MultiOutputQRNNBlockVqc2
import copy
data_path = 'waveforms2500.pk'
data_num =2500
train_rate = 0.6



class Logger(object):
    def __init__(self, file_path: str = "./Default.log"):
        self.terminal = sys.stdout
        self.log = open(file_path, "a")
        self.encoding = None

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

class WaveIntegrateDataset(Dataset):
    def __init__(self,path,times=1,reshape = False,num_block=1,index = None,window_size =None):
        self.data = np.load(path,allow_pickle=True)
        self.X = self.data[0]
        self.Y = self.data[1]
        if index is not None:
            self.X =self.X[index]
            self.Y = self.Y[index]
        self.len = self.Y.shape[0]
        self.channel = self.X.shape[-1]
        

        self.X = self.X * 1e5
        # self.X = abs(self.X)
        self.X = np.swapaxes(self.X,1,2)
        if window_size ==None:
            self.feats = self.X.shape[1]
        else:
            self.feats = window_size
            self.X = self.X[:,:,:self.feats]

        self.X = self.X.reshape(self.len,self.channel,num_block,int(self.feats/num_block))
        self.integrate = np.sum(self.X,axis=-1)
        self.times = times
        self.reshape = reshape

    def __len__(self):
        return self.times * self.len

    def __getitem__(self, idx):
        x = self.integrate[idx%self.len]
        if self.reshape:
            x= x.reshape(-1)
        x = torch.tensor(x,dtype=torch.float32)
        y = self.Y[idx%self.len]
        return x,y




def train(dataflow, model, device, optimizer):
    target_all = []
    output_all = []
    
    for batch_idx, (data, target) in enumerate(dataflow):
        inputs = data.to(device)
        targets = target.to(device)

        outputs = model(inputs)

        loss = F.nll_loss(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        target_all.append(targets)
        output_all.append(outputs)

    target_all = torch.cat(target_all, dim=0)
    output_all = torch.cat(output_all, dim=0)

    _, indices = output_all.topk(1, dim=1)
    masks = indices.eq(target_all.view(-1, 1).expand_as(indices))
    size = target_all.shape[0]
    corrects = masks.sum().item()
    accuracy = corrects / size

    return accuracy

def valid_test(dataflow, model, device, qiskit=False):
    target_all = []
    output_all = []
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(dataflow):
            inputs = data.to(device)
            targets = target.to(device)
            outputs = model(inputs)

            target_all.append(targets)
            output_all.append(outputs)

        target_all = torch.cat(target_all, dim=0)
        output_all = torch.cat(output_all, dim=0)

    _, indices = output_all.topk(1, dim=1)
    masks = indices.eq(target_all.view(-1, 1).expand_as(indices))
    size = target_all.shape[0]
    corrects = masks.sum().item()
    accuracy = corrects / size
    loss = F.nll_loss(output_all, target_all).item()
    return accuracy


def main(blocks = 4 , n_vqc_qubit =4,bsize=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--static', action='store_true', help='compute with '
                                                              'static mode')
    parser.add_argument('--wires-per-block', type=int, default=2,
                        help='wires per block int static mode')
    parser.add_argument('--epochs', type=int, default=50,
                        help='number of training epochs')

    args = parser.parse_args()
    random_index = list(range(0,data_num))
    random.shuffle(random_index)
    train_index = random_index[:int(data_num*train_rate)]
    test_index = random_index[int(data_num*train_rate):]

    train_db = WaveIntegrateDataset(data_path,times=1,reshape=False,num_block=blocks,index=train_index,window_size =bsize)
    test_db = WaveIntegrateDataset(data_path,times=1,reshape=False,num_block=blocks,index=test_index,window_size =bsize)
    train_data = DataLoader(train_db, batch_size=128, shuffle=True)
    test_data = DataLoader(test_db, batch_size=128, shuffle=True)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # model = QFCModel(4).to(device)
    # model = QRNN(4).to(device)
    model = QRNNBlockVqc(n_block = blocks,n_states=n_vqc_qubit).to(device)

    n_epochs = args.epochs
    optimizer = optim.Adam(model.parameters(), lr = 0.05)
    scheduler = CosineAnnealingLR(optimizer, T_max=n_epochs)
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [20,30,50], gamma=0.9)
    best_acc =0

    if args.static:
        model.q_layer.static_on(wires_per_block=args.wires_per_block)

    for epoch in range(1, n_epochs + 1):
        # train
        train(train_data, model, device, optimizer)
        acc = valid_test(test_data, model, device)

        if best_acc<acc:
            best_acc = acc
        print(f'Epoch {epoch}: current acc = {acc},best acc = {best_acc}')
        scheduler.step()

if __name__ == '__main__':
    sys.stdout = Logger('./logger/{}_log.txt'.format('block3_xyz_window_size_test'))
    n_features =[4]
    n_vqc_qubits = [4]
    block_sizes = [256]
    for n_feature in n_features:
        for n_vqc_qubit in n_vqc_qubits:
            for block_size in block_sizes:
                print()
                print( 'n_stages :{}, n_vqc_qubit: {}, block_size: {} '.format(n_feature,n_vqc_qubit,block_size))
                main(n_feature,n_vqc_qubit,block_size)
