#include "configuration.h"
#include "neuralNetwork.h"
#include "readHeader.h"
#include "stdio.h"

int main(int argc, char * argv[]) {
	sConfig conf;
	sDataset train_dataset, test_dataset;
	sNetworkWithSigmoid network;

	if(configuration("config.txt", &conf)) 
		return 1;

	train_dataset.dataset = read_mnist_images(conf.train_data_path, 
			train_dataset.number_of_images, train_dataset.image_size);
	train_dataset.datalabel = read_mnist_labels(conf.train_label_path,
			train_dataset.number_of_labels);

	test_dataset.dataset = read_mnist_images(conf.test_data_path, 
			test_dataset.number_of_images, test_dataset.image_size);
	test_dataset.datalabel = read_mnist_labels(conf.test_label_path,
			test_dataset.number_of_labels);

	initilizationForMnist(&network, conf);

	train(network, train_dataset, conf.learnRate, conf.numEras); 

	printf("mean error on train: %lf\n", meanError(network, train_dataset));
	printf("mean error on test: %lf\n", meanError(network, test_dataset));

	return 0;
}