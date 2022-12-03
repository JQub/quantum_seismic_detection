from __future__ import print_function

import torchquantum as tq


class QLayer22(tq.QuantumModule):
    def __init__(self):
        super().__init__()
        self.n_wires = 4
        self.layer_indexs  = dict()
        self.RYs1 = tq.Op1QAllLayer(op=tq.RY, n_wires=4,has_params=True,trainable=True)
        self.RXs1 = tq.Op1QAllLayer(op=tq.RX, n_wires=4,has_params=True,trainable=True)
        self.RZs1 = tq.Op1QAllLayer(op=tq.RZ, n_wires=4,has_params=True,trainable=True)

        self.CRYs1 = tq.Op2QAllLayer(op=tq.CRY,n_wires=4,has_params=True,trainable=True,circular =True) #Op2QAllLayer
        self.CRXs1 = tq.Op2QAllLayer(op=tq.CRX,n_wires=4,has_params=True,trainable=True,circular =True)
        self.CRZs2 = tq.Op2QAllLayer(op=tq.CRZ,n_wires=2,has_params=True,trainable=True,circular =True)
        # self.hadmard = tq.Hadamard(n_wires=4,wires=[0, 1,2,3])
    @tq.static_support
    def forward(self, q_device: tq.QuantumDevice):
        self.q_device = q_device
        # self.hadmard(self.q_device)
        # add dense trainable gates
        self.CRZs2(self.q_device)
        self.RYs1(self.q_device)
        self.CRYs1(self.q_device)
        self.RXs1(self.q_device)
        self.RZs1(self.q_device)
        self.CRXs1(self.q_device)
        # self.hadmard(self.q_device)

class QLayer18(tq.QuantumModule):
    def __init__(self):
        super().__init__()
        self.n_wires = 4
        self.layer_indexs  = dict()
        self.RYs1 = tq.Op1QAllLayer(op=tq.RY, n_wires=4,has_params=True,trainable=True)
        self.RXs1 = tq.Op1QAllLayer(op=tq.RX, n_wires=4,has_params=True,trainable=True)
        self.RZs1 = tq.Op1QAllLayer(op=tq.RZ, n_wires=4,has_params=True,trainable=True)

        self.CRYs1 = tq.Op2QAllLayer(op=tq.CRY,n_wires=4,has_params=True,trainable=True,circular =True) #Op2QAllLayer
        # self.CRXs1 = tq.Op2QAllLayer(op=tq.CRX,n_wires=4,has_params=True,trainable=True,circular =True)
        self.CRZs2 = tq.Op2QAllLayer(op=tq.CRZ,n_wires=2,has_params=True,trainable=True,circular =True)
        # self.hadmard = tq.Hadamard(n_wires=4,wires=[0, 1,2,3])
    @tq.static_support
    def forward(self, q_device: tq.QuantumDevice):
        self.q_device = q_device
        # self.hadmard(self.q_device)
        # add dense trainable gates
        self.CRZs2(self.q_device)
        self.RYs1(self.q_device)
        self.CRYs1(self.q_device)
        self.RXs1(self.q_device)
        self.RZs1(self.q_device)
        # self.CRXs1(self.q_device)
        # self.hadmard(self.q_device)


class QLayer30(tq.QuantumModule):
    def __init__(self):
        super().__init__()
        self.n_wires = 4
        self.layer_indexs  = dict()
        self.RYs1 = tq.Op1QAllLayer(op=tq.RY, n_wires=4,has_params=True,trainable=True)
        self.RXs1 = tq.Op1QAllLayer(op=tq.RX, n_wires=4,has_params=True,trainable=True)
        self.RZs1 = tq.Op1QAllLayer(op=tq.RZ, n_wires=4,has_params=True,trainable=True)
        self.RYs2 = tq.Op1QAllLayer(op=tq.RY, n_wires=4,has_params=True,trainable=True)
        self.RXs2 = tq.Op1QAllLayer(op=tq.RX, n_wires=4,has_params=True,trainable=True)

        self.CRYs1 = tq.Op2QAllLayer(op=tq.CRY,n_wires=4,has_params=True,trainable=True,circular =True) #Op2QAllLayer
        self.CRXs1 = tq.Op2QAllLayer(op=tq.CRX,n_wires=4,has_params=True,trainable=True,circular =True)
        self.CRZs2 = tq.Op2QAllLayer(op=tq.CRZ,n_wires=2,has_params=True,trainable=True,circular =True)
        # self.hadmard = tq.Hadamard(n_wires=4,wires=[0, 1,2,3])
    @tq.static_support
    def forward(self, q_device: tq.QuantumDevice):
        self.q_device = q_device
        # self.hadmard(self.q_device)
        # add dense trainable gates
        self.CRZs2(self.q_device)
        self.RYs1(self.q_device)
        self.CRYs1(self.q_device)
        self.RXs1(self.q_device)
        self.RZs1(self.q_device)
        self.CRXs1(self.q_device)

        self.RYs2(self.q_device)
        self.RXs2(self.q_device)
        # self.hadmard(self.q_device)

class QLayer1(tq.QuantumModule):
    def __init__(self,n_wire =4):
        super().__init__()
        self.n_wires = n_wire
        self.layer_indexs  = dict()
        self.RYs1 = tq.Op1QAllLayer(op=tq.RY, n_wires=self.n_wires,has_params=True,trainable=True)
        self.RXs1 = tq.Op1QAllLayer(op=tq.RX, n_wires=self.n_wires,has_params=True,trainable=True)
        self.RYs2 = tq.Op1QAllLayer(op=tq.RY, n_wires=self.n_wires,has_params=True,trainable=True)
        # self.RXs2 = tq.Op1QAllLayer(op=tq.RX, n_wires=self.n_wires,has_params=True,trainable=True)
        # self.RYs3 = tq.Op1QAllLayer(op=tq.RY, n_wires=4,has_params=True,trainable=True)
        # self.RXs3 = tq.Op1QAllLayer(op=tq.RX, n_wires=4,has_params=True,trainable=True)
        self.RZs3 = tq.Op1QAllLayer(op=tq.RZ, n_wires=4,has_params=True,trainable=True)

        self.CRZs1 = tq.Op2QAllLayer(op=tq.CRZ,n_wires=self.n_wires,has_params=True,trainable=True,circular =False) #Op2QAllLayer
        # self.CRXs1 = tq.Op2QAllLayer(op=tq.CRX,n_wires=self.n_wires,has_params=True,trainable=True,circular =True)
        # self.CRZs3 = tq.Op2QAllLayer(op=tq.CRZ,n_wires=3,has_params=True,trainable=True,circular =True)
        # self.hadmard = tq.Hadamard(n_wires=4,wires=[0, 1,2,3])
    @tq.static_support
    def forward(self, q_device: tq.QuantumDevice):
        self.q_device = q_device
        # self.hadmard(self.q_device)
        # add dense trainable gates
        # self.CRZs2(self.q_device)
        self.RYs1(self.q_device)
        self.CRZs1(self.q_device)
        self.RYs2(self.q_device)
        self.RXs1(self.q_device)
        self.CRZs1(self.q_device)

        # self.CRXs1(self.q_device)
        # self.RXs2(self.q_device)



        # self.CRZs3(self.q_device)
        # self.RYs3(self.q_device)
        # self.RXs3(self.q_device)
        
        # self.hadmard(self.q_device)


class QLayer4(tq.QuantumModule):
    def __init__(self,n_wire =4):
        super().__init__()
        self.n_wires = n_wire
        self.layer_indexs  = dict()
        self.RYs1 = tq.Op1QAllLayer(op=tq.RY, n_wires=self.n_wires,has_params=True,trainable=True)
        self.RXs1 = tq.Op1QAllLayer(op=tq.RX, n_wires=self.n_wires,has_params=True,trainable=True)
        self.RZs1 = tq.Op1QAllLayer(op=tq.RZ, n_wires=self.n_wires,has_params=True,trainable=True)
        self.RYs2 = tq.Op1QAllLayer(op=tq.RY, n_wires=self.n_wires,has_params=True,trainable=True)
        self.RXs2 = tq.Op1QAllLayer(op=tq.RX, n_wires=self.n_wires,has_params=True,trainable=True)
        self.RZs2 = tq.Op1QAllLayer(op=tq.RZ, n_wires=self.n_wires,has_params=True,trainable=True)
        # self.RYs3 = tq.Op1QAllLayer(op=tq.RY, n_wires=4,has_params=True,trainable=True)
        # self.RXs3 = tq.Op1QAllLayer(op=tq.RX, n_wires=4,has_params=True,trainable=True)
        # self.RZs3 = tq.Op1QAllLayer(op=tq.RZ, n_wires=4,has_params=True,trainable=True)

        self.CRYs1 = tq.Op2QAllLayer(op=tq.CRY,n_wires=self.n_wires,has_params=True,trainable=True,circular =True) #Op2QAllLayer
        self.CRXs1 = tq.Op2QAllLayer(op=tq.CRX,n_wires=self.n_wires,has_params=True,trainable=True,circular =True)
        self.CRZs1 = tq.Op2QAllLayer(op=tq.CRZ,n_wires=self.n_wires,has_params=True,trainable=True,circular =True)
        self.CRZs2 = tq.Op2QAllLayer(op=tq.CRZ,n_wires=self.n_wires,has_params=True,trainable=True,circular =True)
        # self.CRZs3 = tq.Op2QAllLayer(op=tq.CRZ,n_wires=3,has_params=True,trainable=True,circular =True)
        # self.hadmard = tq.Hadamard(n_wires=4,wires=[0, 1,2,3])
    @tq.static_support
    def forward(self, q_device: tq.QuantumDevice):
        self.q_device = q_device
        # self.hadmard(self.q_device)
        # add dense trainable gates
        # self.CRZs2(self.q_device)
        self.RYs1(self.q_device)
        self.CRYs1(self.q_device)
        self.RYs2(self.q_device)

        self.RXs1(self.q_device)
        self.CRXs1(self.q_device)
        self.RXs2(self.q_device)

        self.RZs1(self.q_device)
        self.CRZs1(self.q_device)
        
        self.RZs2(self.q_device)
        self.CRZs2(self.q_device)

        # self.CRZs3(self.q_device)
        # self.RYs3(self.q_device)
        # self.RXs3(self.q_device)
        
        # self.hadmard(self.q_device)


class VQC3_block(tq.QuantumModule):
    def __init__(self,n_wire =4):
        super().__init__()
        self.n_wires = n_wire
        self.layer_indexs  = dict()
        self.RYs1 = tq.Op1QAllLayer(op=tq.RY, n_wires=self.n_wires,has_params=True,trainable=True)
        self.RXs1 = tq.Op1QAllLayer(op=tq.RX, n_wires=self.n_wires,has_params=True,trainable=True)
        self.RZs1 = tq.Op1QAllLayer(op=tq.RZ, n_wires=self.n_wires,has_params=True,trainable=True)
        # self.RYs2 = tq.Op1QAllLayer(op=tq.RY, n_wires=self.n_wires,has_params=True,trainable=True)
        # self.RXs2 = tq.Op1QAllLayer(op=tq.RX, n_wires=self.n_wires,has_params=True,trainable=True)
        # self.RZs2 = tq.Op1QAllLayer(op=tq.RZ, n_wires=self.n_wires,has_params=True,trainable=True)
        # self.RYs3 = tq.Op1QAllLayer(op=tq.RY, n_wires=4,has_params=True,trainable=True)
        # self.RXs3 = tq.Op1QAllLayer(op=tq.RX, n_wires=4,has_params=True,trainable=True)
        # self.RZs3 = tq.Op1QAllLayer(op=tq.RZ, n_wires=4,has_params=True,trainable=True)

        self.CRYs1 = tq.Op2QAllLayer(op=tq.CRY,n_wires=self.n_wires,has_params=True,trainable=True,circular =True) #Op2QAllLayer
        self.CRXs1 = tq.Op2QAllLayer(op=tq.CRX,n_wires=self.n_wires,has_params=True,trainable=True,circular =True)
        self.CRZs1 = tq.Op2QAllLayer(op=tq.CRZ,n_wires=self.n_wires,has_params=True,trainable=True,circular =True)
        # self.CRZs2 = tq.Op2QAllLayer(op=tq.CRZ,n_wires=self.n_wires,has_params=True,trainable=True,circular =True)
        # self.CRZs3 = tq.Op2QAllLayer(op=tq.CRZ,n_wires=3,has_params=True,trainable=True,circular =True)
        # self.hadmard = tq.Hadamard(n_wires=4,wires=[0, 1,2,3])
    @tq.static_support
    def forward(self, q_device: tq.QuantumDevice):
        self.q_device = q_device
        # self.hadmard(self.q_device)
        # add dense trainable gates
        # self.CRZs2(self.q_device)
        self.RYs1(self.q_device)
        self.CRYs1(self.q_device)
        # self.RYs2(self.q_device)

        self.RXs1(self.q_device)
        self.CRXs1(self.q_device)
        # self.RXs2(self.q_device)

        self.RZs1(self.q_device)
        self.CRZs1(self.q_device)
        
        # self.RZs2(self.q_device)
        # self.CRZs2(self.q_device)

        # self.CRZs3(self.q_device)
        # self.RYs3(self.q_device)
        # self.RXs3(self.q_device)
        
        # self.hadmard(self.q_device)



class QLayerManmade(tq.QuantumModule):
    def __init__(self,n_wire =4):
        super().__init__()
        self.n_wires = n_wire
        self.layer_indexs  = dict()
        # self.RYs1 = tq.Op1QAllLayer(op=tq.RY, n_wires=self.n_wires,has_params=True,trainable=True)
        self.RXs1 = tq.Op1QAllLayer(op=tq.RX, n_wires=self.n_wires,has_params=True,trainable=True)
        self.RZs1 = tq.Op1QAllLayer(op=tq.RZ, n_wires=self.n_wires,has_params=True,trainable=True)
        # self.RYs2 = tq.Op1QAllLayer(op=tq.RY, n_wires=self.n_wires,has_params=True,trainable=True)
        self.RXs2 = tq.Op1QAllLayer(op=tq.RX, n_wires=self.n_wires,has_params=True,trainable=True)
        self.RZs2 = tq.Op1QAllLayer(op=tq.RZ, n_wires=self.n_wires,has_params=True,trainable=True)
        # self.RYs3 = tq.Op1QAllLayer(op=tq.RY, n_wires=4,has_params=True,trainable=True)
        # self.RXs3 = tq.Op1QAllLayer(op=tq.RX, n_wires=4,has_params=True,trainable=True)
        # self.RZs3 = tq.Op1QAllLayer(op=tq.RZ, n_wires=4,has_params=True,trainable=True)

        # self.CRYs1 = tq.Op2QAllLayer(op=tq.CRY,n_wires=self.n_wires,has_params=True,trainable=True,circular =True) #Op2QAllLayer
        self.CRXs1 = tq.Op2QAllLayer(op=tq.CRX,n_wires=self.n_wires,has_params=True,trainable=True,circular =True)
        self.CRZs1 = tq.Op2QAllLayer(op=tq.CRZ,n_wires=self.n_wires,has_params=True,trainable=True,circular =True)
        # self.CRZs3 = tq.Op2QAllLayer(op=tq.CRZ,n_wires=3,has_params=True,trainable=True,circular =True)
        # self.hadmard = tq.Hadamard(n_wires=4,wires=[0, 1,2,3])
    @tq.static_support
    def forward(self, q_device: tq.QuantumDevice):
        self.q_device = q_device
        # self.hadmard(self.q_device)
        # add dense trainable gates
        # self.CRZs2(self.q_device)
        # self.RYs1(self.q_device)
        # self.CRYs1(self.q_device)
        # self.RYs2(self.q_device)

        self.RXs1(self.q_device)
        self.CRXs1(self.q_device)
        self.RXs2(self.q_device)

        self.RZs1(self.q_device)
        self.CRZs1(self.q_device)
        self.RZs2(self.q_device)


class QLayer_6q(tq.QuantumModule):
    def __init__(self):
        super().__init__()
        self.n_wires = 6
        self.layer_indexs  = dict()
        self.RYs1 = tq.Op1QAllLayer(op=tq.RY, n_wires=6,has_params=True,trainable=True)
        self.RXs1 = tq.Op1QAllLayer(op=tq.RX, n_wires=6,has_params=True,trainable=True)
        self.RZs1 = tq.Op1QAllLayer(op=tq.RZ, n_wires=6,has_params=True,trainable=True)
        self.RYs2 = tq.Op1QAllLayer(op=tq.RY, n_wires=6,has_params=True,trainable=True)
        self.RXs2 = tq.Op1QAllLayer(op=tq.RX, n_wires=6,has_params=True,trainable=True)
        self.RZs2 = tq.Op1QAllLayer(op=tq.RZ, n_wires=6,has_params=True,trainable=True)
        # self.RYs3 = tq.Op1QAllLayer(op=tq.RY, n_wires=4,has_params=True,trainable=True)
        # self.RXs3 = tq.Op1QAllLayer(op=tq.RX, n_wires=4,has_params=True,trainable=True)
        # self.RZs3 = tq.Op1QAllLayer(op=tq.RZ, n_wires=4,has_params=True,trainable=True)

        self.CRYs1 = tq.Op2QAllLayer(op=tq.CRY,n_wires=6,has_params=True,trainable=True,circular =True) #Op2QAllLayer
        self.CRXs1 = tq.Op2QAllLayer(op=tq.CRX,n_wires=6,has_params=True,trainable=True,circular =True)
        self.CRZs1 = tq.Op2QAllLayer(op=tq.CRZ,n_wires=6,has_params=True,trainable=True,circular =True)
        self.CRZs2 = tq.Op2QAllLayer(op=tq.CRZ,n_wires=2,has_params=True,trainable=True,circular =True)
        # self.hadmard = tq.Hadamard(n_wires=4,wires=[0, 1,2,3,4,5])
    @tq.static_support
    def forward(self, q_device: tq.QuantumDevice):
        self.q_device = q_device
        # self.hadmard(self.q_device)
        # add dense trainable gates
        self.CRZs2(self.q_device)
        self.RYs1(self.q_device)
        self.CRYs1(self.q_device)
        self.RXs1(self.q_device)
        self.RZs1(self.q_device)
        self.CRXs1(self.q_device)

        self.RYs2(self.q_device)
        self.RXs2(self.q_device)
        self.CRZs1(self.q_device)
        self.RZs2(self.q_device)

        # self.hadmard(self.q_device)


class QLayer_big(tq.QuantumModule):
    def __init__(self):
        super().__init__()
        self.n_wires = 4
        self.layer_indexs  = dict()
        self.RYs1 = tq.Op1QAllLayer(op=tq.RY, n_wires=4,has_params=True,trainable=True)
        self.RXs1 = tq.Op1QAllLayer(op=tq.RX, n_wires=4,has_params=True,trainable=True)
        self.RZs1 = tq.Op1QAllLayer(op=tq.RZ, n_wires=4,has_params=True,trainable=True)
        self.RYs2 = tq.Op1QAllLayer(op=tq.RY, n_wires=4,has_params=True,trainable=True)
        self.RXs2 = tq.Op1QAllLayer(op=tq.RX, n_wires=4,has_params=True,trainable=True)
        self.RZs2 = tq.Op1QAllLayer(op=tq.RZ, n_wires=4,has_params=True,trainable=True)
        self.RYs3 = tq.Op1QAllLayer(op=tq.RY, n_wires=4,has_params=True,trainable=True)
        self.RXs3 = tq.Op1QAllLayer(op=tq.RX, n_wires=4,has_params=True,trainable=True)

        self.CRYs1 = tq.Op2QAllLayer(op=tq.CRY,n_wires=4,has_params=True,trainable=True,circular =True) #Op2QAllLayer
        self.CRXs1 = tq.Op2QAllLayer(op=tq.CRX,n_wires=4,has_params=True,trainable=True,circular =True)
        self.CRZs1 = tq.Op2QAllLayer(op=tq.CRZ,n_wires=4,has_params=True,trainable=True,circular =True)
        self.CRZs2 = tq.Op2QAllLayer(op=tq.CRZ,n_wires=2,has_params=True,trainable=True,circular =True)
        # self.CRZs3 = tq.Op2QAllLayer(op=tq.CRZ,n_wires=3,has_params=True,trainable=True,circular =True)

        self.RZs4 = tq.Op1QAllLayer(op=tq.RZ, n_wires=4,has_params=True,trainable=True)
        # self.RYs4 = tq.Op1QAllLayer(op=tq.RY, n_wires=4,has_params=True,trainable=True)
        # self.RXs4 = tq.Op1QAllLayer(op=tq.RX, n_wires=4,has_params=True,trainable=True)
        # self.CRYs4 = tq.Op2QAllLayer(op=tq.CRY,n_wires=4,has_params=True,trainable=True,circular =True) #Op2QAllLayer
        # self.CRXs4 = tq.Op2QAllLayer(op=tq.CRX,n_wires=4,has_params=True,trainable=True,circular =True)
        # self.CRZs4 = tq.Op2QAllLayer(op=tq.CRZ,n_wires=4,has_params=True,trainable=True,circular =True)
        # self.RZs5 = tq.Op1QAllLayer(op=tq.RZ, n_wires=4,has_params=True,trainable=True)
        # self.RYs5 = tq.Op1QAllLayer(op=tq.RY, n_wires=4,has_params=True,trainable=True)
        # self.RXs5 = tq.Op1QAllLayer(op=tq.RX, n_wires=4,has_params=True,trainable=True)



        # self.hadmard = tq.Hadamard(n_wires=4,wires=[0, 1,2,3])
    @tq.static_support
    def forward(self, q_device: tq.QuantumDevice):
        self.q_device = q_device
        # self.hadmard(self.q_device)
        # add dense trainable gates
        self.CRZs2(self.q_device)
        self.RYs1(self.q_device)
        self.CRYs1(self.q_device)
        self.RXs1(self.q_device)
        self.RZs1(self.q_device)
        self.CRXs1(self.q_device)

        self.RYs2(self.q_device)
        self.RXs2(self.q_device)
        self.CRZs1(self.q_device)
        self.RZs2(self.q_device)


        # self.CRZs3(self.q_device)
        self.RYs3(self.q_device)
        self.RXs3(self.q_device)

        self.RZs4 (self.q_device)
        # self.RYs4 (self.q_device)
        # self.RXs4 (self.q_device)
        # self.CRYs4(self.q_device) #Op2QAllLayer
        # self.CRXs4(self.q_device)
        # self.CRZs4(self.q_device)
        # self.RZs5 (self.q_device)
        # self.RYs5 (self.q_device)
        # self.RXs5 (self.q_device)

        # self.hadmard(self.q_device)

class QLayer_big2(tq.QuantumModule):
    def __init__(self):
        super().__init__()
        self.n_wires = 4
        self.layer_indexs  = dict()
        self.RYs1 = tq.Op1QAllLayer(op=tq.RY, n_wires=4,has_params=True,trainable=True)
        self.RXs1 = tq.Op1QAllLayer(op=tq.RX, n_wires=4,has_params=True,trainable=True)
        self.RZs1 = tq.Op1QAllLayer(op=tq.RZ, n_wires=4,has_params=True,trainable=True)
        self.RYs2 = tq.Op1QAllLayer(op=tq.RY, n_wires=4,has_params=True,trainable=True)
        self.RXs2 = tq.Op1QAllLayer(op=tq.RX, n_wires=4,has_params=True,trainable=True)
        self.RZs2 = tq.Op1QAllLayer(op=tq.RZ, n_wires=4,has_params=True,trainable=True)
        self.RYs3 = tq.Op1QAllLayer(op=tq.RY, n_wires=4,has_params=True,trainable=True)
        self.RXs3 = tq.Op1QAllLayer(op=tq.RX, n_wires=4,has_params=True,trainable=True)

        self.CRYs1 = tq.Op2QAllLayer(op=tq.CRY,n_wires=4,has_params=True,trainable=True,circular =True) #Op2QAllLayer
        self.CRXs1 = tq.Op2QAllLayer(op=tq.CRX,n_wires=4,has_params=True,trainable=True,circular =True)
        self.CRZs1 = tq.Op2QAllLayer(op=tq.CRZ,n_wires=4,has_params=True,trainable=True,circular =True)
        self.CRZs2 = tq.Op2QAllLayer(op=tq.CRZ,n_wires=2,has_params=True,trainable=True,circular =True)
        # self.CRZs3 = tq.Op2QAllLayer(op=tq.CRZ,n_wires=3,has_params=True,trainable=True,circular =True)

        self.RZs4 = tq.Op1QAllLayer(op=tq.RZ, n_wires=4,has_params=True,trainable=True)
        self.RYs4 = tq.Op1QAllLayer(op=tq.RY, n_wires=4,has_params=True,trainable=True)
        self.RXs4 = tq.Op1QAllLayer(op=tq.RX, n_wires=4,has_params=True,trainable=True)
        # self.CRYs4 = tq.Op2QAllLayer(op=tq.CRY,n_wires=4,has_params=True,trainable=True,circular =True) #Op2QAllLayer
        self.CRXs4 = tq.Op2QAllLayer(op=tq.CRX,n_wires=4,has_params=True,trainable=True,circular =True)
        self.CRZs4 = tq.Op2QAllLayer(op=tq.CRZ,n_wires=4,has_params=True,trainable=True,circular =True)
        self.RZs5 = tq.Op1QAllLayer(op=tq.RZ, n_wires=4,has_params=True,trainable=True)
        # self.RYs5 = tq.Op1QAllLayer(op=tq.RY, n_wires=4,has_params=True,trainable=True)
        # self.RXs5 = tq.Op1QAllLayer(op=tq.RX, n_wires=4,has_params=True,trainable=True)



        # self.hadmard = tq.Hadamard(n_wires=4,wires=[0, 1,2,3])
    @tq.static_support
    def forward(self, q_device: tq.QuantumDevice):
        self.q_device = q_device
        # self.hadmard(self.q_device)
        # add dense trainable gates
        self.CRZs2(self.q_device)
        self.RYs1(self.q_device)
        self.CRYs1(self.q_device)
        self.RXs1(self.q_device)
        self.RZs1(self.q_device)
        self.CRXs1(self.q_device)

        self.RYs2(self.q_device)
        self.RXs2(self.q_device)
        self.CRZs1(self.q_device)
        self.RZs2(self.q_device)


        # self.CRZs3(self.q_device)
        self.RYs3(self.q_device)
        self.RXs3(self.q_device)

        self.RZs4 (self.q_device)
        self.RYs4 (self.q_device)
        self.RXs4 (self.q_device)
        # self.CRYs4(self.q_device) #Op2QAllLayer
        self.CRXs4(self.q_device)
        self.CRZs4(self.q_device)
        self.RZs5 (self.q_device)
        # self.RYs5 (self.q_device)
        # self.RXs5 (self.q_device)

        # self.hadmard(self.q_device)

class QLayer_mini(tq.QuantumModule):
    def __init__(self):
        super().__init__()
        self.n_wires = 3

        ops = [tq.RX(has_params=True,trainable=True,wires=[0]),tq.RY(has_params=True,trainable=True,wires=[1]),
        tq.RZ(has_params=True,trainable=True,wires=[2]),tq.RY(has_params=True,trainable=True,wires=[0]),
        tq.CRX(has_params=True,trainable=True,wires=[1,2]),tq.CRZ(has_params=True,trainable=True,wires=[0,1]),
        tq.RX(has_params=True,trainable=True,wires=[2]),tq.RX(has_params=True,trainable=True,wires=[2]),
        tq.RY(has_params=True,trainable=True,wires=[1]),tq.RZ(has_params=True,trainable=True,wires=[0]),
        tq.RX(has_params=True,trainable=True,wires=[0]),tq.RY(has_params=True,trainable=True,wires=[1]),
        tq.RZ(has_params=True,trainable=True,wires=[2])]
        self.op_modules = tq.QuantumModuleFromOps(ops)


    @tq.static_support
    def forward(self, q_device: tq.QuantumDevice):
        self.q_device = q_device

        self.op_modules(self.q_device)


class QLayer_very_mini(tq.QuantumModule):
    def __init__(self):
        super().__init__()
        self.n_wires = 2

        ops = [tq.RX(has_params=True,trainable=True,wires=[0]),tq.RY(has_params=True,trainable=True,wires=[1]),
        tq.RZ(has_params=True,trainable=True,wires=[1]),tq.RY(has_params=True,trainable=True,wires=[0]),
        tq.CRZ(has_params=True,trainable=True,wires=[0,1]),tq.RZ(has_params=True,trainable=True,wires=[0]),
        tq.RX(has_params=True,trainable=True,wires=[0]),tq.RX(has_params=True,trainable=True,wires=[1]),
        tq.CRX(has_params=True,trainable=True,wires=[1,0]),tq.RY(has_params=True,trainable=True,wires=[1])]
        self.op_modules = tq.QuantumModuleFromOps(ops)


    @tq.static_support
    def forward(self, q_device: tq.QuantumDevice):
        self.q_device = q_device

        self.op_modules(self.q_device)

class QLayer_very_mini15(tq.QuantumModule):
    def __init__(self):
        super().__init__()
        self.n_wires = 2

        ops = [tq.RX(has_params=True,trainable=True,wires=[0]),tq.RY(has_params=True,trainable=True,wires=[1]),
        tq.RZ(has_params=True,trainable=True,wires=[1]),tq.RY(has_params=True,trainable=True,wires=[0]),
        tq.CRZ(has_params=True,trainable=True,wires=[0,1]),tq.RZ(has_params=True,trainable=True,wires=[0]),
        tq.RX(has_params=True,trainable=True,wires=[0]),tq.RX(has_params=True,trainable=True,wires=[1]),
        tq.CRX(has_params=True,trainable=True,wires=[1,0]),tq.RY(has_params=True,trainable=True,wires=[1])]
        self.op_modules = tq.QuantumModuleFromOps(ops)


    @tq.static_support
    def forward(self, q_device: tq.QuantumDevice):
        self.q_device = q_device

        self.op_modules(self.q_device)