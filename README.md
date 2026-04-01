# Privacy-Preserving Decentralized Secure Aggregation in Cross-Silo Federated Learning

This repository is based on an implementation from [Shaoxiong Ji's PyTorch Implementation of Federated Learning](http://doi.org/10.5281/zenodo.4321561), which is a reproduction of the paper [Communication-Efficient Learning of Deep Networks from Decentralized Data](https://arxiv.org/abs/1602.05629). Our contributions include the implementation of a secure aggregation scheme and a simplified SVFL protocol, both of which are detailed in our research.



## Requirements
- python>=3.6  
- pytorch>=0.4  
- cryptography package: `pip install cryptography`

## Running the Code

### Secure Aggregation
To run the federated learning with secure aggregation, use the following command:
```bash
python main_fed.py --dataset mnist --iid --secure_aggregation
```

See the available arguments in options.py for further customization.

### SVFL Protocol
Before running the SVFL protocol, you must first compute the necessary global parameters by running:
```bash
python g_calculation.py
```

## Acknowledgements
The base code was derived from [Shaoxiong Ji's PyTorch Implementation of Federated Learning](http://doi.org/10.5281/zenodo.4321561), with our contributions focusing on implementing secure aggregation and the SVFL protocol.
