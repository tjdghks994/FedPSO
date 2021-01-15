## FedPSO: Federated Learning Using Particle Swarm Optimization to Reduce Communication Costs

## Abstract
Federated learning is a learning method that collects only learned models on a server to ensure data privacy. This method does not collect data on the server but instead proceeds with data directly from distributed clients. Because federated learning clients often have limited communication bandwidth, communication between servers and clients should be optimized to improve performance. Federated learning clients often use Wi-Fi and have to communicate in unstable network environments. However, as existing federated learning aggregation algorithms transmit and receive a large amount of weights, accuracy is significantly reduced in unstable network environments. 
In this study, we propose the algorithm using particle swarm optimization algorithm instead of FedAvg, which updates the global model by collecting weights of learned models that were mainly used in federated learning. The algorithm is named as federated particle swarm optimization (FedPSO), and we increase its robustness in unstable network environments by transmitting score values rather than large weights. Thus, we propose a FedPSO, a global model update algorithm with improved network communication performance, by changing the form of the data that clients transmit to servers. This study showed that applying FedPSO significantly reduced the amount of data used in network communication and improved the accuracy of the global model by an average of 9.47%. Moreover, it showed an improvement in loss of accuracy by approximately 4% in experiments on an unstable network.

## Figure

