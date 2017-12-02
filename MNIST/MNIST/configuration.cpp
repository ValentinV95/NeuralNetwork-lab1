#include "configuration.h"
#include "stdlib.h"
#include "stdio.h"

int configuration(char * full_path_file, sConfig * conf) {
	FILE * file = fopen(full_path_file, "r");
	if(file != NULL) {
		if(fscanf(file,"train_data_path = %s\n", conf->train_data_path) != 1) return 1;
		if(fscanf(file,"train_label_path = %s\n", conf->train_label_path) != 1) return 1;
		if(fscanf(file,"test_data_path = %s\n", conf->test_data_path) != 1) return 1;
		if(fscanf(file,"test_label_path = %s\n", conf->test_label_path) != 1) return 1;
		if(fscanf(file,"number_hidden_neurons = %i\n", &conf->number_hidden_neurons) != 1)
			conf->number_hidden_neurons = 400;
		if(fscanf(file,"learn_rate = %lf\n", &conf->learnRate) != 1)
			conf->learnRate = 0.5;
		if(fscanf(file,"number_eras = %i", &conf->numEras) != 1)
			conf->numEras = 100;

	} else {
        printf("Cannot open config file!");
		return 1;
    }

	return 0;
}