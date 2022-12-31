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
from wave_qmodel import QRNNBlockVqcAmpEnc
import copy


class PulseInputDataset(Dataset):
    def __init__(self,path,input_name,index = None):
        self.X = np.load(path + input_name +'.npy')
        self.Y = np.load(path + 'label.npy')
        if index is not None:
            self.X =self.X[index]
            self.Y = self.Y[index]
        self.len = self.Y.shape[0]
        

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        x = self.X[idx]
        x = torch.tensor(x,dtype=torch.float32)
        y = self.Y[idx]
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


def main(blocks = 2 , n_vqc_qubit =6):
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

    train_db = PulseInputDataset('pulse_input/','mats_abs2x10',index=train_index)
    test_db = PulseInputDataset('pulse_input/','mats_abs2x10',index=test_index)
    train_data = DataLoader(train_db, batch_size=128, shuffle=True)
    test_data = DataLoader(test_db, batch_size=128, shuffle=True)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    model = QRNNBlockVqcAmpEnc(n_block = blocks,n_states=n_vqc_qubit).to(device)

    n_epochs = args.epochs
    optimizer = optim.Adam(model.parameters(), lr=0.01)
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
    sys.stdout = Logger('./logger/{}_log.txt'.format('ttttest'))
    main()
                
