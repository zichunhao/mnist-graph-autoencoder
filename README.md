# MNIST Graph Autoencoder
## Main References
The implementations of graph neural network in [models/GraphNet.py](https://github.com/zichunhao/mnist_graph_autoencoder/blob/master/models/GraphNet.py) and [MNISTGraphDataset](https://github.com/zichunhao/mnist_graph_autoencoder/blob/master/MNISTGraphDataset.py) were adapted from Raghav's [Graph Generative Adversarial Networks for Sparse Data Generation](https://github.com/rkansal47/graph-gan) Project.

The implementation of the chamfer loss function, `chamfer_loss` in [utils/loss.py](https://github.com/zichunhao/mnist_graph_autoencoder/blob/master/utils/loss.py) was from Steven Tsan's [Particle Graph Autoencoders for Anomaly Detection](https://github.com/stsan9/AnomalyDetection4Jets/tree/emd) project.

## Training Data
The MNIST dataset on which the model was trained is in `.csv` format downloaded from [here](https://github.com/pjreddie/mnist-csv-png).
