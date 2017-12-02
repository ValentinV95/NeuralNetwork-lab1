#ifndef NETWORK
#define NETWORK
#include "structures.h"

void initilizationForMnist(sNetworkWithSigmoid * network, sConfig conf);

double crossEntropy(sNetworkWithSigmoid network, sDataset set);

void train(sNetworkWithSigmoid network, sDataset trainset, double learnRate, int numEras);

double meanError(sNetworkWithSigmoid network, sDataset set);

#endif

