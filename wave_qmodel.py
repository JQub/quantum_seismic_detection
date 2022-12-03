from audioop import reverse
from turtle import position
import torch
import torch.nn.functional as F
import torch.optim as optim
import argparse

import pytorch_quantum.torchquantum.functional as tqf
import random
from pytorch_quantum.examples.core.datasets import MNIST
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset,DataLoader
import sys
import numpy as np

import pytorch_quantum.torchquantum as tq
from qlayers import QLayer_block1,QLayer_block2,QLayer_block3

def combination(n,c,com=1,limit=0,per=[]):
    for pos in range(limit,n):
        t = per + [pos]
        if len(set(t)) == len(t):
            if len(t) == c:
                    yield [pos,]
            else:
                    for result in combination(n,c,com,com*pos, per + [pos,]):
                            yield [pos,] + result

class QRNNBlock(tq.QuantumModule):
    def __init__(self,n_states,n_work_stages=10,n_IOs=3):
        super().__init__()
        self.n_IOs = n_IOs
        self.n_states = n_states
        self.n_wires = self.n_IOs + self.n_states
        self.work_stages = n_work_stages
        #input stage 可训练参数
        self.INs = torch.nn.ModuleList()
        for i in range(self.n_states):
            CN = torch.nn.ModuleList()
            CN.append(tq.RY(has_params=True,trainable=True,wires=0))
            CN.append(tq.CRY(has_params=True,trainable=True,wires=[0,self.n_IOs+i]))
            CN.append(tq.CRY(has_params=True,trainable=True,wires=[1,self.n_IOs+i]))
            CN.append(tq.CRY(has_params=True,trainable=True,wires=[2,self.n_IOs+i]))
            CN.append(tq.MultiCRY(has_params=True,trainable=True,n_wires=3,wires=[0,1,self.n_IOs+i]))
            CN.append(tq.MultiCRY(has_params=True,trainable=True,n_wires=3,wires=[0,2,self.n_IOs+i]))
            CN.append(tq.MultiCRY(has_params=True,trainable=True,n_wires=3,wires=[1,2,self.n_IOs+i]))
            self.INs.extend(CN)


        
        #work stage 可训练参数
        self.Rys = torch.nn.ModuleList()
        for i in range(self.n_states):
            self.Rys.append(tq.RY(has_params=True,trainable=True,wires=n_IOs+i))
        self.random_vqc = tq.RandomLayer(n_ops=n_work_stages,wires=list(range(self.n_wires)))
        self.Ws = torch.nn.ModuleList()
        for i in range(self.n_states):
            CN = torch.nn.ModuleList()
            rotate_qubit = self.n_IOs+i
            control_qubits = list(range(self.n_wires))
            control_qubits.remove(rotate_qubit)
            # print(control_qubits)
            CN.append(tq.RY(has_params=True,trainable=True,wires=rotate_qubit)) #theta0
            for cq in control_qubits:#theta1
                rand = np.random.randint(2,size=1)
                # if rand == 0:
                #     CN.append(tq.CRX(has_params=True,trainable=True,wires=[cq,rotate_qubit]))
                # else:
                CN.append(tq.CRY(has_params=True,trainable=True,wires=[cq,rotate_qubit]))
                
            cq2_lists = list()
            for res in  combination(len(control_qubits),2):#theta2
                cq2_lists.append(res)
            # print(cq2_lists)
            for cq2 in cq2_lists:
                CN.append(tq.MultiCRY(has_params=True,trainable=True,n_wires=3,wires=[control_qubits[cq2[0]],control_qubits[cq2[1]],rotate_qubit]))

            cq3_lists = list()
            for res in  combination(len(control_qubits),3):#theta3
                cq3_lists.append(res)
            # print(cq2_lists)
            for cq2 in cq3_lists:
                CN.append(tq.MultiCRY(has_params=True,trainable=True,n_wires=4,wires=[control_qubits[cq2[0]],control_qubits[cq2[1]],control_qubits[cq2[2]],rotate_qubit]))

    
            # cq6_lists = list()
            # for res in  combination(len(control_qubits),5):#theta5
            #     cq6_lists.append(res)
            # # print(cq2_lists)
            # for cq2 in cq6_lists:
            #     qubit_list = list()
            #     for q in cq2:
            #         qubit_list.append(control_qubits[q])
            #     qubit_list.append(rotate_qubit)
            #     # print(qubit_list)
            #     CN.append(tq.MultiCRY(has_params=True,trainable=True,n_wires=6,wires= qubit_list))
            self.Ws.extend(CN)

    @tq.static_support
    def forward(self, q_device: tq.QuantumDevice):
        self.q_device = q_device
        for layer in self.INs:
            layer(self.q_device)
        for layer in self.Rys:
            layer(self.q_device)
        for layer in self.Ws:
            layer(self.q_device)
        # self.random_vqc(self.q_device)

class QRNN(tq.QuantumModule):
    def __init__(self,n_feat,n_states=4,n_ios=3):
        super().__init__()
        self.n_feat = n_feat
        self.n_states = n_states
        self.n_ios = n_ios
        self.n_wires = self.n_ios + self.n_states

        self.qrnnBlock = QRNNBlock(n_states)
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)

        self.encoder_rx = tq.GeneralEncoder(tq.encoder_op_list_name_dict['3_rx'])
        self.encoder_ry = tq.GeneralEncoder(tq.encoder_op_list_name_dict['3_ry'])
        self.encoder_rz = tq.GeneralEncoder(tq.encoder_op_list_name_dict['3_rz'])

        self.Hs = torch.nn.ModuleList()
        for i in range(self.n_states):
            self.Hs.append(tq.Hadamard(wires=n_ios+i))

        self.measure = tq.Measure(tq.PauliZ,wires=list(range(self.n_ios,self.n_ios + self.n_states)))

    def forward(self, x, use_qiskit=False):
        bsz = x.shape[0]
        n_feat = x.shape[2]
        x = x.view(bsz,-1, n_feat)
        self.reset = True

        for i in range(x.shape[2]):
            encode_temp = x[:,:,i]
            self.encoder_ry(self.q_device, encode_temp,self.reset)
            self.reset = False
            self.encoder_rx(self.q_device, encode_temp,self.reset)
            self.encoder_rz(self.q_device, encode_temp,self.reset)

            self.qrnnBlock(self.q_device)

        x = self.measure(self.q_device)
        # print(x.shape)
        x = x.reshape(bsz, 2, -1).sum(-1).squeeze()
        x = F.log_softmax(x, dim=1)

        return x






class QFCModel(tq.QuantumModule):
    def __init__(self,n_feat):
        super().__init__()
        self.n_wires = 6
        self.n_feat = n_feat
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)

        self.encoder_rx = tq.GeneralEncoder(tq.encoder_op_list_name_dict['3_rx'])
        self.encoder_ry = tq.GeneralEncoder(tq.encoder_op_list_name_dict['3_ry'])
        self.encoder_rz = tq.GeneralEncoder(tq.encoder_op_list_name_dict['3_rz'])

        self.reset_flag = True
        self.q_layer = QLayer_block1(self.n_wires)
        self.q_layers = torch.nn.Sequential(
                self.q_layer,
                *[self.q_layer for i in range(self.n_feat-1)],
        )
        self.measure = tq.Measure(tq.PauliZ,wires=[0,1,2,3,4,5])

    def forward(self, x, use_qiskit=False):
        # print(x.shape)
        bsz = x.shape[0]
        n_feat = x.shape[2]
        x = x.view(bsz,-1, n_feat)
        # print(x.shape)
        if use_qiskit:
            x = self.qiskit_processor.process_parameterized(
                self.q_device, self.encoder, self.q_layer, self.measure, x)
        else:
            self.reset_flag = True
            for i in range(n_feat):
                encode_temp = x[:,:,i]
                # print(i)
                self.encoder_rx(self.q_device, encode_temp,reset_flag = self.reset_flag)
                self.reset_flag = False
                self.encoder_ry(self.q_device, encode_temp,reset_flag = self.reset_flag)
                self.encoder_rz(self.q_device, encode_temp,reset_flag = self.reset_flag)

                # print(self.q_device.states.shape)
                self.q_layer(self.q_device)

            x = self.measure(self.q_device)

        x = x.reshape(bsz, 2, -1).sum(-1).squeeze()
        x = F.log_softmax(x, dim=1)

        return x

def genEncoder(input=[],wires=[],func=[]):
    encoder_list =[]
    for i in range(len(input)):
        en = dict()
        en['input_idx']= list([input[i]])
        en['func']= func[i]
        en['wires']= list([wires[i]])
        encoder_list.append(en)
    return encoder_list

class QRNNBlockVqc(tq.QuantumModule):
    def __init__(self,n_block,n_states=4,n_ios=3):
        super().__init__()
        self.n_block = n_block
        self.n_states = n_states
        self.n_ios = n_ios
        self.n_wires = self.n_states + self.n_ios

        self.q_layer = QLayer_block3(n_states)
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)

        v_input = [0,0,0,1,1,1,2,2,2]
        v_wires = [n_states+0,n_states+0,n_states+0,n_states+1,n_states+1,n_states+1,n_states+2,n_states+2,n_states+2]
        # v_funcs = ['ry','rx','rz','ry','rx','rz','ry','rx','rz']
        v_funcs = ['rx','ry','rz','rx','ry','rz','rx','ry','rz']
        encoder_list = genEncoder(v_input,v_wires,v_funcs)
        self.encoder = tq.GeneralEncoder(encoder_list)

        self.measure = tq.Measure(tq.PauliZ,wires=list(range(0, self.n_states)))

        self.CNOT0 =tq.CNOT(self.q_device, wires=[n_states+0, 0])
        self.CNOT1 =tq.CNOT(self.q_device, wires=[n_states+1, 1])
        self.CNOT2 =tq.CNOT(self.q_device, wires=[n_states+2, 2])

    def forward(self, x, use_qiskit=False):
        # print(x.shape)
        bsz = x.shape[0]
        n_block = x.shape[2]
        x = x.view(bsz,-1, n_block)
        # print(x.shape)
        if use_qiskit:
            x = self.qiskit_processor.process_parameterized(
                self.q_device, self.encoder, self.q_layer, self.measure, x)
        else:
            self.reset_flag = True
            for i in range(n_block):
                encode_temp = x[:,:,i]
                # print(i)
                self.encoder(self.q_device, encode_temp,reset_flag = self.reset_flag)
                self.reset_flag = False

                self.CNOT0(self.q_device)
                self.CNOT1(self.q_device)
                self.CNOT2(self.q_device)

                self.q_layer(self.q_device)

            x = self.measure(self.q_device)


        x = x.reshape(bsz, 2, -1).sum(-1).squeeze()
        x = F.log_softmax(x, dim=1)

        return x

def state_exp(n_bits,state_bits,state):
    bsz = state.shape[0]
    astate =  torch.zeros(bsz,2**n_bits)
    positions = []
    n_states = len(state_bits)
    state_s = '{0:0'+str(n_states)+'b}'
    for i in range(2**n_states):
        state_ss =  list(state_s.format(i))
        state_ss.reverse()
        astate_str = list('0'*n_bits)
        for j in range(n_states):
            astate_str[state_bits[j]] = state_ss[j]
        astate_str.reverse()
        pos = int(''.join([str(elem) for elem in astate_str]),2)
        astate[:,pos] = state [:,i]
    # print(astate)
    return astate

# stat = torch.tensor([[1,2,3,4,5,6,7,8]])
# state_exp(4,[1,2,3],stat)
# sys.exit(0)



class QRNNBlockVqcAmpEnc(tq.QuantumModule):
    def __init__(self,n_block,n_states=4,n_ios=3):
        super().__init__()
        self.n_block = n_block
        self.n_states = n_states
        self.n_ios = n_ios
        self.n_wires = self.n_states + self.n_ios

        self.q_layer1 = QLayer_block3(n_states)
        # self.q_layer2 = QLayer_block1(n_states)
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)


        self.encoder = tq.StateEncoder()

        self.measure = tq.Measure(tq.PauliZ,wires=list(range(0, self.n_states)))

        self.CY1 =tq.CY(self.q_device, wires=[n_states+0, 0])
        self.CY2 =tq.CY(self.q_device, wires=[n_states+1, 1])
        self.CY3 =tq.CY(self.q_device, wires=[n_states+2, 2])
        self.CX1 =tq.CNOT(self.q_device, wires=[n_states+0, 0])
        self.CX2 =tq.CNOT(self.q_device, wires=[n_states+1, 1])
        self.CX3 =tq.CNOT(self.q_device, wires=[n_states+2, 2])
        self.CZ1 =tq.CZ(self.q_device, wires=[n_states+0, 0])
        self.CZ2 =tq.CZ(self.q_device, wires=[n_states+1, 1])
        self.CZ3 =tq.CZ(self.q_device, wires=[n_states+2, 2])


    def forward(self, x, use_qiskit=False):
        # print(x.shape)
        bsz = x.shape[0]
        n_block = x.shape[1]
        if use_qiskit:
            x = self.qiskit_processor.process_parameterized(
                self.q_device, self.encoder, self.q_layer, self.measure, x)
        else:
            for i in range(n_block):
                encode_temp = x[:,i,:]
                # print(encode_temp[0,:])
                encode_temp = state_exp(self.q_device.n_wires,[self.n_states+0,self.n_states+1,self.n_states+2],encode_temp)
                # print(encode_temp[0,:])
                self.encoder(self.q_device, encode_temp)



                self.CX1(self.q_device)
                self.CX2(self.q_device)
                self.CX3(self.q_device)

                self.CY1(self.q_device)
                self.CY2(self.q_device)
                self.CY3(self.q_device)

                self.CZ1(self.q_device)
                self.CZ2(self.q_device)
                self.CZ3(self.q_device)


                self.q_layer1(self.q_device)
                # self.q_layer2(self.q_device)

            x = self.measure(self.q_device)


        x = x.reshape(bsz, 2, -1).sum(-1).squeeze()
        x = F.log_softmax(x, dim=1)

        return x


class MultiOutputQRNNBlockVqc(tq.QuantumModule):
    def __init__(self,n_feat,n_states=4,n_ins=3):
        super().__init__()
        self.n_feat = n_feat
        self.n_states = n_states # the wires of VQC
        self.n_ins = n_ins
        self.n_outs=n_feat *2
        self.n_wires = self.n_states + self.n_ins + self.n_outs

        # 0 -> n_states : vqc
        self.q_layer = QLayer_block3(n_states)
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)

        # n_states -> n_states + n_ins: input
        v_input = [0,0,0,1,1,1,2,2,2]
        v_wires = [n_states+0,n_states+0,n_states+0,n_states+1,n_states+1,n_states+1,n_states+2,n_states+2,n_states+2]
        # v_funcs = ['ry','rx','rz','ry','rx','rz','ry','rx','rz']
        v_funcs = ['rx','ry','rz','rx','ry','rz','rx','ry','rz']
        encoder_list = genEncoder(v_input,v_wires,v_funcs)
        self.encoder = tq.GeneralEncoder(encoder_list)

        # n_states + n_ins -> n_wires : output
        self.measure = tq.Measure(tq.PauliZ,wires=list(range(self.n_states+self.n_ins, self.n_wires)))

        # cx in: from input to vqc
        self.CNOT_IN0 =tq.CNOT(self.q_device, wires=[n_states+0, 0])
        self.CNOT_IN1 =tq.CNOT(self.q_device, wires=[n_states+1, 1])
        self.CNOT_IN2 =tq.CNOT(self.q_device, wires=[n_states+2, 2])

        #cx out: from vqc to output
        self.OUTs = torch.nn.ModuleList()

        # for j in range(self.n_states+self.n_ins, self.n_wires):
        #     for i in range(self.n_states):
        #             CNOT_OUT =tq.CNOT(self.q_device, wires=[i, j])
        #             self.OUTs.append(CNOT_OUT)
        CNOT_OUT =tq.CNOT(self.q_device, wires=[0, self.n_states+self.n_ins+0])
        self.OUTs.append(CNOT_OUT)
        CNOT_OUT =tq.CNOT(self.q_device, wires=[1, self.n_states+self.n_ins+0])
        self.OUTs.append(CNOT_OUT)
        CNOT_OUT =tq.CNOT(self.q_device, wires=[2, self.n_states+self.n_ins+1])
        self.OUTs.append(CNOT_OUT)
        CNOT_OUT =tq.CNOT(self.q_device, wires=[3, self.n_states+self.n_ins+1])
        self.OUTs.append(CNOT_OUT)

        CNOT_OUT =tq.CNOT(self.q_device, wires=[0, self.n_states+self.n_ins+2])
        self.OUTs.append(CNOT_OUT)
        CNOT_OUT =tq.CNOT(self.q_device, wires=[1, self.n_states+self.n_ins+2])
        self.OUTs.append(CNOT_OUT)
        CNOT_OUT =tq.CNOT(self.q_device, wires=[2, self.n_states+self.n_ins+3])
        self.OUTs.append(CNOT_OUT)
        CNOT_OUT =tq.CNOT(self.q_device, wires=[3, self.n_states+self.n_ins+3])
        self.OUTs.append(CNOT_OUT)
        CNOT_OUT =tq.CNOT(self.q_device, wires=[0, self.n_states+self.n_ins+4])
        self.OUTs.append(CNOT_OUT)
        CNOT_OUT =tq.CNOT(self.q_device, wires=[1, self.n_states+self.n_ins+4])
        self.OUTs.append(CNOT_OUT)
        CNOT_OUT =tq.CNOT(self.q_device, wires=[2, self.n_states+self.n_ins+5])
        self.OUTs.append(CNOT_OUT)
        CNOT_OUT =tq.CNOT(self.q_device, wires=[3, self.n_states+self.n_ins+5])
        self.OUTs.append(CNOT_OUT)

    def forward(self, x, use_qiskit=False):
        # print(x.shape)
        bsz = x.shape[0]
        n_block = x.shape[2]
        x = x.view(bsz,-1, n_block)
        # print(x.shape)
        if use_qiskit:
            x = self.qiskit_processor.process_parameterized(
                self.q_device, self.encoder, self.q_layer, self.measure, x)
        else:
            self.reset_flag = True
            for i in range(n_block):
                encode_temp = x[:,:,i]
                self.encoder(self.q_device, encode_temp,reset_flag = self.reset_flag)
                self.reset_flag = False

                self.CNOT_IN0(self.q_device)
                self.CNOT_IN1(self.q_device)
                self.CNOT_IN2(self.q_device)

                self.q_layer(self.q_device)
                for j in range(self.n_states):
                    self.OUTs[i*self.n_states+j](self.q_device)

            x = self.measure(self.q_device)
            # print('='*50)
            # print(x)
            # I = torch.zeros_like(x)
            # x2 = I-x
            # x = torch.cat((x,x2),dim=1)
            x = x.reshape(bsz,self.n_feat,2, -1).sum(-1).squeeze()
            # x = x.reshape(bsz,-1)
            # x = F.log_softmax(x,dim=1)
        return x

class MultiOutputQRNNBlockVqc2(tq.QuantumModule):
    def __init__(self,n_feat,n_states=4,n_ins=3):
        super().__init__()
        self.n_feat = n_feat
        self.n_states = n_states # the wires of VQC
        self.n_ins = n_ins
        self.n_outs=n_feat 
        self.n_wires = self.n_states + self.n_ins + self.n_outs

        # 0 -> n_states : vqc
        self.q_layer = QLayer_block3(n_states)
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)

        # n_states -> n_states + n_ins: input
        v_input = [0,0,0,1,1,1,2,2,2]
        v_wires = [n_states+0,n_states+0,n_states+0,n_states+1,n_states+1,n_states+1,n_states+2,n_states+2,n_states+2]
        # v_funcs = ['ry','rx','rz','ry','rx','rz','ry','rx','rz']
        v_funcs = ['rx','ry','rz','rx','ry','rz','rx','ry','rz']
        encoder_list = genEncoder(v_input,v_wires,v_funcs)
        self.encoder = tq.GeneralEncoder(encoder_list)

        # n_states + n_ins -> n_wires : output
        self.measure = tq.Measure(tq.PauliZ,wires=list(range(self.n_states+self.n_ins, self.n_wires)))

        # cx in: from input to vqc
        self.CNOT_IN0 =tq.CNOT(self.q_device, wires=[n_states+0, 0])
        self.CNOT_IN1 =tq.CNOT(self.q_device, wires=[n_states+1, 1])
        self.CNOT_IN2 =tq.CNOT(self.q_device, wires=[n_states+2, 2])

        #cx out: from vqc to output
        self.OUTs = torch.nn.ModuleList()
        
        # for j in range(self.n_states+self.n_ins, self.n_wires):
        #     for i in range(self.n_states):
        #             CNOT_OUT =tq.CNOT(self.q_device, wires=[i, j])
        #             self.OUTs.append(CNOT_OUT)
        CNOT_OUT =tq.CNOT(self.q_device, wires=[0, self.n_states+self.n_ins+0])
        self.OUTs.append(CNOT_OUT)
        CNOT_OUT =tq.CNOT(self.q_device, wires=[1, self.n_states+self.n_ins+0])
        self.OUTs.append(CNOT_OUT)
        CNOT_OUT =tq.CNOT(self.q_device, wires=[2, self.n_states+self.n_ins+1])
        self.OUTs.append(CNOT_OUT)
        CNOT_OUT =tq.CNOT(self.q_device, wires=[3, self.n_states+self.n_ins+1])
        self.OUTs.append(CNOT_OUT)



    def forward(self, x, use_qiskit=False):
        # print(x.shape)
        bsz = x.shape[0]
        n_block = x.shape[2]
        x = x.view(bsz,-1, n_block)
        # print(x.shape)
        if use_qiskit:
            x = self.qiskit_processor.process_parameterized(
                self.q_device, self.encoder, self.q_layer, self.measure, x)
        else:
            self.reset_flag = True
            for i in range(n_block):
                encode_temp = x[:,:,i]
                # print(i)
                self.encoder(self.q_device, encode_temp,reset_flag = self.reset_flag)
                self.reset_flag = False

                self.CNOT_IN0(self.q_device)
                self.CNOT_IN1(self.q_device)
                self.CNOT_IN2(self.q_device)

                self.q_layer(self.q_device)

            for j in range(self.n_states):
                self.OUTs[j](self.q_device)

            x = self.measure(self.q_device)
            # I = torch.zeros_like(x)
            # x2 = I-x
            # x = torch.cat((x,x2),dim=1)
            # x = x.reshape(bsz,self.n_feat,2, -1).sum(-1).squeeze()
            x = x.reshape(bsz,2, -1).sum(-1).squeeze()
            x = F.log_softmax(x, dim=1)
        return x





class QRNN(tq.QuantumModule):
    def __init__(self,n_feat,n_states=4,n_ios=3):
        super().__init__()
        self.n_feat = n_feat
        self.n_states = n_states
        self.n_ios = n_ios
        self.n_wires = self.n_ios + self.n_states

        self.qrnnBlock = QRNNBlock(n_states)
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        self.encoder_ry = tq.GeneralEncoder(tq.encoder_op_list_name_dict['3_rx'])
        self.encoder_rx = tq.GeneralEncoder(tq.encoder_op_list_name_dict['3_ry'])
        self.encoder_rz = tq.GeneralEncoder(tq.encoder_op_list_name_dict['3_rz'])

        # self.Hs = torch.nn.ModuleList()
        self.measure = tq.Measure(tq.PauliZ,wires=list(range(self.n_ios,self.n_ios + self.n_states)))

    def forward(self, x, use_qiskit=False):
        bsz = x.shape[0]
        n_feat = x.shape[2]
        x = x.view(bsz,-1, n_feat)
        self.reset = True

        for i in range(x.shape[2]):
            encode_temp = x[:,:,i]
            self.encoder_ry(self.q_device, encode_temp,self.reset)
            self.reset = False
            self.encoder_rx(self.q_device, encode_temp,self.reset)
            self.encoder_rz(self.q_device, encode_temp,self.reset)
            self.qrnnBlock(self.q_device)
        x = self.measure(self.q_device)
        x = x.reshape(bsz, 2, -1).sum(-1).squeeze()
        x = F.log_softmax(x, dim=1)

        return x