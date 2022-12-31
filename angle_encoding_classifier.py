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
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        self.encoder = tq.GeneralEncoder(
            tq.encoder_op_list_name_dict['4x4_ryzxy'])

        self.q_layer = self.QLayer(self.n_wires)
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x, use_qiskit=False):
        bsz = x.shape[0]

        if use_qiskit:
            x = self.qiskit_processor.process_parameterized(
                self.q_device, self.encoder, self.q_layer, self.measure, x)
        else:
            self.encoder(self.q_device, x)
            self.q_layer(self.q_device)
            x = self.measure(self.q_device)

        # print(x.shape)
        x = x.reshape(bsz, 2, 2).sum(-1).squeeze()
        x = F.log_softmax(x, dim=1)

        return x

class WaveIntegrateDataset(torch.utils.data.Dataset):
    def __init__(self,index = None,path = 'waveforms1500.npy',times=1,num_block=16,window_size =50):
        self.data = np.load(path,allow_pickle=True).item()
        self.X = self.data['data']
        self.Y = self.data['target']
        if index is not None:
            self.X =self.X[index]
            self.Y = self.Y[index]
        self.channel = self.X.shape[-1]
        self.X = self.X * 1e4
        self.X = abs(self.X)
        self.X = np.swapaxes(self.X,1,2)
        self.X = self.X[:,1,:]
        self.len = num_block *window_size
        self.samples = len(self.Y)
        print(self.X.shape)
        self.X = self.X[:,100:1000]
        self.X = self.X[:,:self.len]
        print(self.X.shape)
        self.X = self.X.reshape(self.samples,num_block,window_size)
        self.integrate = np.sum(self.X,axis=-1)
        self.times = times

    def __len__(self):
        return self.times * self.samples

    def __getitem__(self, idx):
        x = self.integrate[idx%self.samples]
        x = torch.tensor(x,dtype=torch.float32)
        y = self.Y[idx%self.samples]
        return {'data': x, 'target': y}


class Simple2Class(Dataset):
    def __init__(self):
        data_num = 1500
        train_rate =0.9
        random_index = list(range(0,data_num))
        random.shuffle(random_index)
        train_index = random_index[:int(data_num*train_rate)]
        test_index = random_index[int(data_num*train_rate):]
        train_dataset = WaveIntegrateDataset(index = train_index)
        valid_dataset = WaveIntegrateDataset(index = test_index)
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
