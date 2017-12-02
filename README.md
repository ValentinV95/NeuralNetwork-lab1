# NeuralNetwork
Full connected network for detection MNIST dataset

MNIST DATASET: http://yann.lecun.com/exdb/mnist/
VISUAL STUDIO 2012 project

### Parameters in file config.txt:
1: Path to MNIST train-images  
2: Path to MNIST train-labels  
3: Path to MNIST test-images  
4: Path to MNIST test-labels  
5: number hidden neuron
6: learn rate
7: number epochs

### Example config.txt (is in the repository)
train_data_path = train-images.idx3-ubyte
train_label_path = train-labels.idx1-ubyte
test_data_path = t10k-images.idx3-ubyte
test_label_path = t10k-labels.idx1-ubyte
number_hidden_neurons = 300
learn_rate = 0.01
number_eras = 10

### Error with parameters from example:  
Train: 0.015833
Test: 0.0258