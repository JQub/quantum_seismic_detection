import torch
import torch.nn.functional as F
import torch.optim as optim
import argparse

import torchquantum as tq
import torchquantum.functional as tqf

from torchquantum.datasets import MNIST
from torch.optim.lr_scheduler import CosineAnnealingLR

import random
import numpy as np

class QFCModel(tq.QuantumModule):
    class QLayer(tq.QuantumModule):
        def __init__(self):
            super().__init__()
            self.n_wires = 4
            self.random_layer = tq.RandomLayer(n_ops=50,
                                               wires=list(range(self.n_wires)))

            # gates with trainable parameters
            self.rx0 = tq.RX(has_params=True, trainable=True)
            self.ry0 = tq.RY(has_params=True, trainable=True)
            self.rz0 = tq.RZ(has_params=True, trainable=True)
            self.crx0 = tq.CRX(has_params=True, trainable=True)

        @tq.static_support
        def forward(self, q_device: tq.QuantumDevice):
            """
            1. To convert tq QuantumModule to qiskit or run in the static
            model, need to:
                (1) add @tq.static_support before the forward
                (2) make sure to add
                    static=self.static_mode and
                    parent_graph=self.graph
                    to all the tqf functions, such as tqf.hadamard below
            """
            self.q_device = q_device

            self.random_layer(self.q_device)

            # some trainable gates (instantiated ahead of time)
            self.rx0(self.q_device, wires=0)
            self.ry0(self.q_device, wires=1)
            self.rz0(self.q_device, wires=3)
            self.crx0(self.q_device, wires=[0, 2])

            # add some more non-parameterized gates (add on-the-fly)
            tqf.hadamard(self.q_device, wires=3, static=self.static_mode,
                         parent_graph=self.graph)
            tqf.sx(self.q_device, wires=2, static=self.static_mode,
                   parent_graph=self.graph)
            tqf.cnot(self.q_device, wires=[3, 0], static=self.static_mode,
                     parent_graph=self.graph)
            tqf.rx(self.q_device, wires=1, params=torch.tensor([0.1]),
                   static=self.static_mode, parent_graph=self.graph)

    def __init__(self):
        super().__init__()
        self.n_wires = 4
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        self.encoder = tq.GeneralEncoder(
            tq.encoder_op_list_name_dict['4x4_ryzxy'])

        self.q_layer = self.QLayer()
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x, use_qiskit=False):
        bsz = x.shape[0]
        x = F.avg_pool2d(x, 6).view(bsz, 16)

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


def train(dataflow, model, device, optimizer):
    for feed_dict in dataflow['train']:
        inputs = feed_dict['image'].to(device)
        targets = feed_dict['digit'].to(device)

        outputs = model(inputs)
        loss = F.nll_loss(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"loss: {loss.item()}", end='\r')


def valid_test(dataflow, split, model, device, qiskit=False):
    target_all = []
    output_all = []
    with torch.no_grad():
        for feed_dict in dataflow[split]:
            inputs = feed_dict['image'].to(device)
            targets = feed_dict['digit'].to(device)

            outputs = model(inputs, use_qiskit=qiskit)

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

    print(f"{split} set accuracy: {accuracy}")
    print(f"{split} set loss: {loss}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--static', action='store_true', help='compute with '
                                                              'static mode')
    parser.add_argument('--pdb', action='store_true', help='debug with pdb')
    parser.add_argument('--wires-per-block', type=int, default=2,
                        help='wires per block int static mode')
    parser.add_argument('--epochs', type=int, default=5,
                        help='number of training epochs')

    args = parser.parse_args()

    if args.pdb:
        import pdb
        pdb.set_trace()

    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    dataset = MNIST(
        root='./mnist_data',
        train_valid_split_ratio=[0.9, 0.1],
        digits_of_interest=[3, 6],
        n_test_samples=75,
    )
    dataflow = dict()

    for split in dataset:
        sampler = torch.utils.data.RandomSampler(dataset[split])
        dataflow[split] = torch.utils.data.DataLoader(
            dataset[split],
            batch_size=256,
            sampler=sampler,
            num_workers=8,
            pin_memory=True)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    model = QFCModel().to(device)

    n_epochs = args.epochs
    optimizer = optim.Adam(model.parameters(), lr=5e-3, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=n_epochs)

    if args.static:
        # optionally to switch to the static mode, which can bring speedup
        # on training
        model.q_layer.static_on(wires_per_block=args.wires_per_block)

    for epoch in range(1, n_epochs + 1):
        # train
        print(f"Epoch {epoch}:")
        train(dataflow, model, device, optimizer)
        print(optimizer.param_groups[0]['lr'])

        # valid
        valid_test(dataflow, 'valid', model, device)
        scheduler.step()

    # test
    valid_test(dataflow, 'test', model, device, qiskit=False)

    # run on Qiskit simulator and real Quantum Computers
    try:
        from qiskit import IBMQ
        from modified.qiskit_processor import QiskitProcessor
        from qiskit.test.mock import FakeQuito,FakeBelem


        # firstly perform simulate
        print(f"\nTest with Qiskit Simulator")
        backend = FakeBelem()
        processor_simulation = QiskitProcessor(use_real_qc=False, backend= backend)
        model.set_qiskit_processor(processor_simulation)
        valid_test(dataflow, 'test', model, device, qiskit=True)

        # # then try to run on REAL QC
        # backend_name = 'ibmq_belem'
        # print(f"\nTest on Real Quantum Computer {backend_name}")
        # # Please specify your own hub group and project if you have the
        # # IBMQ premium plan to access more machines.
        # processor_real_qc = QiskitProcessor(use_real_qc=True,
        #                                     backend_name=backend_name,
        #                                     hub='ibm-q',
        #                                     group='open',
        #                                     project='main',
        #                                     )
        # model.set_qiskit_processor(processor_real_qc)
        # valid_test(dataflow, 'test', model, device, qiskit=True)
    except ImportError:
        print("Please install qiskit, create an IBM Q Experience Account and "
              "save the account token according to the instruction at "
              "'https://github.com/Qiskit/qiskit-ibmq-provider', "
              "then try again.")


if __name__ == '__main__':
    main()