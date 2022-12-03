import torch
import torchquantum as tq
import numpy as np

from typing import Union, List


def expval(q_device: tq.QuantumDevice,
           wires: Union[int, List[int]],
           observables: Union[tq.Observable, List[tq.Observable]]):

    all_dims = np.arange(q_device.states.dim())
    if isinstance(wires, int):
        wires = [wires]
        observables = [observables]

    # rotation to the desired basis
    for wire, observable in zip(wires, observables):
        for rotation in observable.diagonalizing_gates():
            rotation(q_device, wires=wire)

    states = q_device.states
    # compute magnitude
    state_mag = torch.abs(states) ** 2

    expectations = []
    for wire, observable in zip(wires, observables):
        # compute marginal magnitude
        print(all_dims)
        reduction_dims = np.delete(all_dims, [0, wire + 1])
        print(reduction_dims)
        probs = state_mag.sum(list(reduction_dims))
        res = probs.mv(observable.eigvals.real.to(probs.device))
        expectations.append(res)

    return torch.stack(expectations, dim=-1)



# def expval(q_device: tq.QuantumDevice,
#            wires: Union[int, List[int]],
#            observables: Union[tq.Observable, List[tq.Observable]]):

#     all_dims = np.arange(q_device.states.dim())
#     if isinstance(wires, int):
#         wires = [wires]
#         observables = [observables]

#     # rotation to the desired basis
#     for wire, observable in zip(wires, observables):
#         for rotation in observable.diagonalizing_gates():
#             rotation(q_device, wires=wire)

#     states = q_device.states
#     # compute magnitude
#     state_mag = torch.abs(states) ** 2

#     expectations = []
#     for wire, observable in zip(wires, observables):
#         # compute marginal magnitude
#         reduction_dims = np.delete(all_dims, [0, wire + 1])
#         if reduction_dims.size == 0:
#             probs = state_mag
#         else:
#             probs = state_mag.sum(list(reduction_dims))
#         res = probs.mv(observable.eigvals.real.to(probs.device))
#         expectations.append(res)

#     return torch.stack(expectations, dim=-1)


class MeasureAll(tq.QuantumModule):
    def __init__(self, obs, v_c_reg_mapping=None):
        super().__init__()
        self.obs = obs
        self.v_c_reg_mapping = v_c_reg_mapping

    def forward(self, q_device: tq.QuantumDevice):
        self.q_device = q_device
        x = expval(q_device, list(range(q_device.n_wires)), [self.obs()] *
                   q_device.n_wires)

        if self.v_c_reg_mapping is not None:
            c2v_mapping = self.v_c_reg_mapping['c2v']
            """
            the measurement is not normal order, need permutation 
            """
            perm = []
            for k in range(x.shape[-1]):
                if k in c2v_mapping.keys():
                    perm.append(c2v_mapping[k])
            x = x[:, perm]

        if self.noise_model_tq is not None and \
                self.noise_model_tq.is_add_noise:
            return self.noise_model_tq.apply_readout_error(x)
        else:
            return x

    def set_v_c_reg_mapping(self, mapping):
        self.v_c_reg_mapping = mapping

class Measure(tq.QuantumModule):
    def __init__(self, obs,wires , v_c_reg_mapping=None):
        super().__init__()
        self.obs = obs
        self.v_c_reg_mapping = v_c_reg_mapping
        self.wires = wires

    def forward(self, q_device: tq.QuantumDevice):
        self.q_device = q_device
        x = expval(q_device, self.wires, [self.obs()] *
                   len(self.wires))

        if self.v_c_reg_mapping is not None:
            c2v_mapping = self.v_c_reg_mapping['c2v']
            """
            the measurement is not normal order, need permutation 
            """
            perm = []
            for k in range(x.shape[-1]):
                if k in c2v_mapping.keys():
                    perm.append(c2v_mapping[k])
            x = x[:, perm]

        if self.noise_model_tq is not None and \
                self.noise_model_tq.is_add_noise:
            return self.noise_model_tq.apply_readout_error(x)
        else:
            return x

    def set_v_c_reg_mapping(self, mapping):
        self.v_c_reg_mapping = mapping
        
class MeasureMultipleTimes(tq.QuantumModule):
    """
    obs list:
    list of dict: example
    [{'wires': [0, 2, 3, 1], 'observables': ['x', 'y', 'z', 'i']
    },
    {'wires': [0, 2, 3, 1], 'observables': ['x', 'y', 'z', 'i']
    },
    ]
    """
    def __init__(self, obs_list, v_c_reg_mapping=None):
        super().__init__()
        self.obs_list = obs_list
        self.v_c_reg_mapping = v_c_reg_mapping

    def forward(self, q_device: tq.QuantumDevice):
        self.q_device = q_device
        res_all = []

        for layer in self.obs_list:
            # create a new q device for each time of measurement
            q_device_new = tq.QuantumDevice(n_wires=q_device.n_wires)
            q_device_new.clone_states(existing_states=q_device.states)
            q_device_new.state = q_device.state

            observables = []
            for wire in range(q_device.n_wires):
                observables.append(tq.I())

            for wire, observable in zip(layer['wires'], layer['observables']):
                observables[wire] = tq.op_name_dict[observable]()

            res = expval(q_device_new, wires=list(range(q_device.n_wires)),
                         observables=observables)

            if self.v_c_reg_mapping is not None:
                c2v_mapping = self.v_c_reg_mapping['c2v']
                """
                the measurement is not normal order, need permutation 
                """
                perm = []
                for k in range(res.shape[-1]):
                    if k in c2v_mapping.keys():
                        perm.append(c2v_mapping[k])
                res = res[:, perm]
            res_all.append(res)

        return torch.cat(res_all)

    def set_v_c_reg_mapping(self, mapping):
        self.v_c_reg_mapping = mapping


class MeasureMultiPauliSum(tq.QuantumModule):
    """
    similar to qiskit.opflow PauliSumOp
    obs list:
    list of dict: example
    [{'wires': [0, 2, 3, 1],
    'observables': ['x', 'y', 'z', 'i'],
    'coefficient': [1, 0.5, 0.4, 0.3]
    },
    {'wires': [0, 2, 3, 1],
    'observables': ['x', 'y', 'z', 'i'],
    'coefficient': [1, 0.5, 0.4, 0.3]
    },
    ]
    """
    def __init__(self, obs_list, v_c_reg_mapping=None):
        super().__init__()
        self.obs_list = obs_list
        self.v_c_reg_mapping = v_c_reg_mapping
        self.measure_multiple_times = MeasureMultipleTimes(
            obs_list=obs_list,
            v_c_reg_mapping=v_c_reg_mapping
        )

    def forward(self, q_device: tq.QuantumDevice):
        res_all = self.measure_multiple_times(q_device)

        return res_all.sum(-1)
