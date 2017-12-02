#include "neuralNetwork.h"
#include "stdio.h"
#include "stdlib.h"
#include "math.h"
#include "omp.h"

void initilizationForMnist(sNetworkWithSigmoid * network, sConfig conf) {
	network->number_input_neurons = 28 * 28;
	network->number_hidden_neurons = conf.number_hidden_neurons;
	network->number_output_neurons = 10;
	network->input = (double *)malloc(sizeof(double) * 
				network->number_input_neurons);
	network->weight_input_hidden = (double *)malloc(sizeof(double) * 
				network->number_input_neurons * network->number_hidden_neurons);
	network->hidden = (double *)malloc(sizeof(double) * 
				network->number_hidden_neurons);
	network->hiddenBias = (double *)malloc(sizeof(double) * 
				network->number_hidden_neurons);
	network->weight_hidden_output = (double *)malloc(sizeof(double) * 
				network->number_hidden_neurons * network->number_output_neurons);
	network->output = (double *)malloc(sizeof(double) * 
				network->number_output_neurons);
	network->outputBias = (double *)malloc(sizeof(double) * 
				network->number_output_neurons);

	
	network->helpGradientOut = (double *)malloc(sizeof(double) * 
				network->number_output_neurons);
	network->helpGradientHidden = (double *)malloc(sizeof(double) * 
				network->number_hidden_neurons);

	srand(42);

	for (int i = 0; i < network->number_input_neurons * network->number_hidden_neurons; i++) {
		network->weight_input_hidden[i] = ((double)rand())/RAND_MAX/1000;
	}

	for (int i = 0; i < network->number_hidden_neurons; i++) {
		network->hiddenBias[i] = ((double)rand())/RAND_MAX/1000;
	}
	
	for (int i = 0; i < network->number_hidden_neurons * network->number_output_neurons; i++) {
		network->weight_hidden_output[i] = ((double)rand())/RAND_MAX/1000;
	}

	for (int i = 0; i < network->number_output_neurons; i++) {
		network->outputBias[i] = ((double)rand())/RAND_MAX/1000;
	}
}

double sigmoid(double * previous, int size, double * weight, double bias) {
	double sum = 0.0;
	for (int i = 0; i < size; i++) {
		sum += previous[i] * weight [i];
	}
	sum += bias;
	if (sum > 10) return 0.0;
	if (sum < -10) return 1.0;
	return 1.0/(1.0 + exp(-sum));
}

void softmax(double * input, int size_input, double * weight, double * bias, double * output,
						int size_output) {
	double max = - 0x7fefffffffffffff;
	for (int i = 0; i < size_output; i++) {
		output[i] = 0.0;
		for (int j = 0; j < size_input; j++) {
			output[i] += weight[i * size_input + j] * input[j];
		}
		output[i] += bias[i];
		if(max < output[i]) {
			max = output[i];
		}
	}
	double norm = 0.0;
	for (int i = 0; i < size_output; i++) {
		output[i] = exp(output[i] - max);
		norm += output[i];
	}

	for (int i = 0; i < size_output; i++) {
		output[i] /= norm;
	}
}

void forwardSigmoid(sNetworkWithSigmoid network, uchar * input, int image_size) {
	for (int i = 0; i < image_size; i++) {
		network.input[i] = (double)input[i]/255.0;
	}

#pragma omp parallel for
	for (int i = 0; i < network.number_hidden_neurons; i++) {
		network.hidden[i] = sigmoid(network.input, network.number_input_neurons,
					&network.weight_input_hidden[i * network.number_input_neurons],
					network.hiddenBias[i]);
	}

	softmax(network.hidden, network.number_hidden_neurons, network.weight_hidden_output,
				network.outputBias, network.output, network.number_output_neurons);
}

void backwardSigmoid(sNetworkWithSigmoid network, int output, double learnRate) {
	for (int i = 0; i < network.number_output_neurons; i ++) { 
		network.helpGradientOut[i] = -network.output[i];
	}
	network.helpGradientOut[output] += 1;

	
#pragma omp parallel for
	for (int i = 0; i < network.number_hidden_neurons; i++) {
		double ddx = -(network.hidden[i] * network.hidden[i] - network.hidden[i]);
		double sum = 0.0;
		for (int j = 0; j < network.number_output_neurons; j++) {
			sum += network.weight_hidden_output[i + j * network.number_hidden_neurons] * network.helpGradientOut[j]; 
		} 
		network.helpGradientHidden[i] = ddx * sum;
	}


#pragma omp parallel for
	for (int i = 0; i < network.number_output_neurons; i++) {
		for (int j = 0; j < network.number_hidden_neurons; j++) {
			network.weight_hidden_output[i * network.number_hidden_neurons + j] += learnRate *
						network.helpGradientOut[i] * network.hidden[j];
		}
	}

	for (int i = 0; i < network.number_output_neurons; i++) {
		network.outputBias[i] += learnRate * network.helpGradientOut[i];
	}

#pragma omp parallel for
	for (int i = 0; i < network.number_hidden_neurons; i++) {	
		for (int j = 0; j < network.number_input_neurons; j++) {
			network.weight_input_hidden[i * network.number_input_neurons + j] += learnRate *
						network.helpGradientHidden[i] * network.input[j];
		}
	}

	for (int i = 0; i < network.number_hidden_neurons; i++) {
		network.hiddenBias[i] += learnRate * network.helpGradientHidden[i];
	}
}

double crossEntropy(sNetworkWithSigmoid network, sDataset set) {
	double error = 0.0;
	for (int i = 0; i < set.number_of_images; i++) {
		forwardSigmoid(network, &set.dataset[i * set.image_size], set.image_size);
		
		error -= log(network.output[set.datalabel[i]]);
	}
	error /= set.number_of_images;

	return error;
}

void train(sNetworkWithSigmoid network, sDataset trainset, double learnRate, int numEras) {

	int * shuf = (int *)malloc(sizeof(int) * trainset.number_of_images);
	for(int i = 0; i < trainset.number_of_images; i++) {
		shuf[i] = i;
	}

	for(int eras = 0; eras < numEras; eras++) {
		for(int i = 0; i < trainset.number_of_images; i++) {
			int index = ((double)rand()/RAND_MAX) * (trainset.number_of_images - 1);
			int tmp = shuf[i];
			shuf[i] = shuf[index];
			shuf[index] = tmp;
		}
		for(int i = 0; i < trainset.number_of_images; i++) {
			int k = shuf[i];
			forwardSigmoid(network, &trainset.dataset[k * trainset.image_size], trainset.image_size);
			backwardSigmoid(network, (int)trainset.datalabel[k], learnRate);
		}

		printf("mean error on train after %i: %lf\n", eras + 1, meanError(network, trainset));
	}
}

double meanError(sNetworkWithSigmoid network, sDataset set) {
	double error = 0.0;
	for (int i = 0; i < set.number_of_images; i++) {
		forwardSigmoid(network, &set.dataset[i * set.image_size], set.image_size);
		
		int k = 0;
		double max = network.output[0];
		for (int j = 1; j < network.number_output_neurons; j++) {
			if(network.output[j] > max) {
				k = j;
				max = network.output[j];
			}
		}
		if(k != (int)set.datalabel[i])
			error += 1.0;
	}
	error /= set.number_of_images;

	return error;
}