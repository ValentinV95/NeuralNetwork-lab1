#ifndef STRUCT
#define STRUCT

typedef unsigned char uchar;
struct sConfig {
	int number_hidden_neurons;
	char train_data_path[200];
	char train_label_path[200];
	char test_data_path[200];
	char test_label_path[200];

	double learnRate;
	int numEras;
};

struct sDataset {
	int number_of_images;
	int number_of_labels;
	int image_size;
	uchar * dataset;
	uchar * datalabel;
};

struct sNetworkWithSigmoid {
	int number_input_neurons;
	int number_hidden_neurons;
	int number_output_neurons;
	double * input;
	double * weight_input_hidden;
	double * hidden;
	double * hiddenBias;
	double * weight_hidden_output;
	double * output;
	double * outputBias;

	double * helpGradientOut;
	double * helpGradientHidden;
};

#endif
