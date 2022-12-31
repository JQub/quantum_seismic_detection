import torch
import torch.nn.functional as F
import torch.optim as optim
import argparse

import torchquantum as tq
import torchquantum.functional as tqf
import random
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset
from torchpack.datasets.dataset import Dataset
import numpy as np
from q_layers import QLayer4
import sys





class QFCModel(tq.QuantumModule):

    class QLayer(tq.QuantumModule):
        def __init__(self,wire):
            super().__init__()
            self.n_wires = wire
            self.layer1 = QLayer4()
            self.layer2 = QLayer4()


        @tq.static_support
        def forward(self, q_device: tq.QuantumDevice):
            self.q_device = q_device

            self.layer1(self.q_device)
            self.layer2(self.q_device)

    def __init__(self):
        super().__init__()
        self.n_wires = 4
        self.q_layer = self.QLayer(self.n_wires)
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        self.encoder = tq.StateEncoder()
        self.measure = tq.MeasureAll(tq.PauliZ)


    def forward(self, x, use_qiskit=False):
        # print(x.shape)
        bsz = x.shape[0]
        if use_qiskit:
            x = self.qiskit_processor.process_parameterized(
                self.q_device, self.encoder, self.q_layer, self.measure, x)
        else:
            self.encoder(self.q_device, x)
            self.q_layer(self.q_device)
            x = self.measure(self.q_device)

        x = x.reshape(bsz, 2, 2).sum(-1).squeeze()
        x = F.log_softmax(x, dim=1)
        return x



class PulseInputDataset(Dataset):
    def __init__(self,index = None,input_path = 'pulse_input/state_mats_abs1x16.npy',label_path = 'waveforms1500.npy'):
        self.X = np.load(input_path)[:,0,:16]
        self.Y = np.load(label_path,allow_pickle=True).item()['target']
        if index is not None:
            self.X =self.X[index]
            self.Y = self.Y[index]
        self.len = self.Y.shape[0]
    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        x = self.X[idx]
        x = torch.tensor(x)
        y = self.Y[idx]
        return {'data': x, 'target': y}

class Simple2Class(Dataset):
    def __init__(self):
        data_num = 1500
        train_rate =0.9
        random_index = list(range(0,data_num))
        random.shuffle(random_index)
        train_index = random_index[:int(data_num*train_rate)]
        test_index = random_index[int(data_num*train_rate):]
        train_dataset = PulseInputDataset(index = train_index)
        valid_dataset = PulseInputDataset(index = test_index)
        datasets = {'train': train_dataset, 'valid': valid_dataset, 'test': valid_dataset}
        super().__init__(datasets)



def train(dataflow,model, optimizer, device = torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    for feed_dict in dataflow['train']:
        inputs = feed_dict['data'].to(device)
        targets = feed_dict['target'].to(device)

        outputs = model(inputs)
        loss = F.nll_loss(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"loss: {loss.item()}", end='\r')



def valid(dataflow, split, model, device, qiskit=False):
    target_all = []
    output_all = []
    with torch.no_grad():
        for feed_dict in dataflow[split]:
            inputs = feed_dict['data'].to(device)
            targets = feed_dict['target'].to(device)

            outputs = model(inputs, use_qiskit=qiskit)

            target_all.append(targets)
            output_all.append(outputs)
        target_all = torch.cat(target_all, dim=0)
        output_all = torch.cat(output_all, dim=0)

    _, indices = output_all.topk(1, dim=1)
    masks = indices.eq(target_all.view(-1, 1).expand_as(indices))
    size = target_all.shape[0]
    corrects = masks.sum().item()
    accuracy = corrects *1.0/ size
    return accuracy




def main():

    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    dataset = Simple2Class()
    dataflow = dict()

    for split in dataset:
        sampler = torch.utils.data.RandomSampler(dataset[split])
        dataflow[split] = torch.utils.data.DataLoader(
            dataset[split],
            batch_size=64,
            sampler=sampler,
            num_workers=8,
            pin_memory=True)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    model = QFCModel().to(device)

    n_epochs = args.epochs
    optimizer = optim.Adam(model.parameters(), lr = 0.005)
    scheduler = CosineAnnealingLR(optimizer, T_max=n_epochs)
    best_accuracy =0.0000

    # if args.static:
    #     model.q_layer.static_on(wires_per_block=args.wires_per_block)
    for epoch in range(1, n_epochs + 1):
        # train
        print(f"Epoch {epoch}:")
        train( dataflow,model, optimizer, device)
        acc = valid(dataflow, 'test', model, device, qiskit=False)
        scheduler.step()

        if acc - best_accuracy >0:
            best_accuracy = acc
        print("accuracy: {}, best accuracy: {}".format(acc,best_accuracy))
        
parser = argparse.ArgumentParser()
# parser.add_argument('--static', action='store_true', help='compute with static mode')
parser.add_argument('--epochs', type=int, default=15)

args = parser.parse_args()


if __name__ == '__main__':

    main()
