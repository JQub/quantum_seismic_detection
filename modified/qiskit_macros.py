import torchquantum as tq

QISKIT_INCOMPATIBLE_OPS = [
    tq.Rot,
    tq.MultiRZ,
    tq.CRot,
]

QISKIT_INCOMPATIBLE_FUNC_NAMES = [
    'rot',
    'multirz',
    'crot',
]

IBMQ_NAMES = [
    'ibmqx2',
    'ibmq_16_melbourne',
    'ibmq_5_yorktown',
    'ibmq_armonk',
    'ibmq_athens',
    'ibmq_belem',
    'ibmq_bogota',
    'ibmq_casablanca',
    'ibmq_dublin',
    'ibmq_guadalupe',
    'ibmq_jakarta',
    'ibmq_kolkata',
    'ibmq_lima',
    'ibmq_manhattan',
    'ibmq_manila',
    'ibmq_montreal',
    'ibmq_mumbai',
    'ibmq_paris',
    'ibmq_qasm_simulator',
    'ibmq_quito',
    'ibmq_rome',
    'ibmq_santiago',
    'ibmq_sydney',
    'ibmq_toronto',
    'simulator_extended_stabilizer',
    'simulator_mps',
    'simulator_stabilizer',
    'simulator_statevector',
    'ibm_auckland',
    'ibm_cairo',
    'ibm_geneva',
    'ibm_hanoi',
    'ibm_ithaca',
    'ibm_lagos',
    'ibm_nairobi',
    'ibm_oslo',
    'ibm_peekskill',
    'ibm_perth',
    'ibm_washington',
    ]

IBMQ_FAKEBACKENDS =[
    'fake_valencia'
]

            # "qasm_simulator_py": "qasm_simulator",
            # "statevector_simulator_py": "statevector_simulator",
            # "unitary_simulator_py": "unitary_simulator",
            # "local_qasm_simulator_py": "qasm_simulator",
            # "local_statevector_simulator_py": "statevector_simulator",
            # "local_unitary_simulator_py": "unitary_simulator",
            # "local_unitary_simulator": "unitary_simulator",

IBMQ_PNAMES = [
    'FakeArmonk',
    'FakeBogota'
    'FakeQuito',
]