# Seismic Detection using Quantum Neural Networks

## Introduction

A synergistic team consisting of researchers from LANL and George Mason University (GMU) has successfully integrated intelligence in the quantum sensor network by using Quantum Neural Network (QNN), such that the acquired data (e.g., seismic data) can be processed in-situ. 
The team took earthquake detection as a vehicle and built the end-to-end framework to evaluate the efficacy of intelligent sensing by encoding seismic data using amplititude, angnle and pulse, processing seismic data using QNN, and optimizing QNN architecture. One key outcome is a quantum neural network compression technique that can reduce quantum circuit length for multiple times while maintaining prediction accuracy, which was published at the 41st IEEE/ACM Conference on Computer-Aided Design. This work has been invited as tutorial talks at ACM/IEEE DAC'22, QuantumWeek'22, ESWEEK'22, and ICCAD'22.

This repo end-to-end gives examples of quantum learning on seismic data. 
Our codes are based on Qiskit APIs and Torch-Quantum.

## An end-to-end example of VQC classifier for seismic detection


### Datasets

We extract 1500 samples of the earthquake detection dataset from FDSN  \cite{fsdn}. Each sample has a positive or negative label. We utilize 90\% and 10\% samples for training and testing, respectively.

### Encoding


```bash
#install:
pip install torchquantum

#angle encoding:
python angle_encoding_classifier.py

#amplitude encoding:
python amplitude_encoding_classifier.py

#pulse encoding:
python generate_pulse_encoding_state.py
python pulse_encoding_classifier.py

```

### Learning

We adopt 2 repeats of a VQC block (4RY +4CRY + 4RY +4RX +4CRX +4RX + 4RZ + 4CRZ +4RZ + 4CRZ) as the model.

## Results

## Simulation Results

|  Encoding   | Accuracy  |
|  ----  | ----  |
| Angle Encoding  | 82% |
| Amplitude Encoding  | 85.3% |
| Pulse Encoding  | 86.3% |


<!-- ### Simulation Results -->

<!-- ### Results on IBM Quantum -->

## Publication

[Quantum Neural Network Compression](https://arxiv.org/abs/2207.01578); 

**Zhirui Hu**, Peiyan Dong, Zhepeng Wang, Youzuo Lin, Yanzhi Wang, Weiwen Jiang

Accepted by IEEE/ACM International Conference On Computer-Aided Design (ICCAD), 2022.


## Collaborators
<img decoding="async" src="logo/lanl.png" width="20%"> 
<img decoding="async" src="logo/mason.png" width="15%">


