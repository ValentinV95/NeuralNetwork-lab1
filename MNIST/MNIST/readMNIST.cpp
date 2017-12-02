#include "readHeader.h"
#include "stdlib.h"
#include "stdio.h"

int reverseInt (int i) {
    unsigned char c1, c2, c3, c4;

    c1 = i & 255;
    c2 = (i >> 8) & 255;
    c3 = (i >> 16) & 255;
    c4 = (i >> 24) & 255;

    return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
}

uchar* read_mnist_images(char * full_path, int& number_of_images, int& image_size) {

    FILE * file = fopen(full_path, "rb");

    if(file != NULL) {
        int magic_number = 0, n_rows = 0, n_cols = 0;

        fread((char *)&magic_number, sizeof(magic_number), 1, file);
        magic_number = reverseInt(magic_number);

        if(magic_number != 2051){
			printf("Invalid MNIST image file!");
			return NULL;
		}

        fread((char *)&number_of_images, sizeof(number_of_images), 1, file);
		number_of_images = reverseInt(number_of_images);
        fread((char *)&n_rows, sizeof(n_rows), 1, file);
		n_rows = reverseInt(n_rows);
        fread((char *)&n_cols, sizeof(n_cols), 1, file);
		n_cols = reverseInt(n_cols);

        image_size = n_rows * n_cols;

        uchar * _dataset = (uchar *)malloc(sizeof(uchar) * number_of_images * image_size);
        for(int i = 0; i < number_of_images; i++) {
            fread((char *)&(_dataset[i * image_size]), sizeof(char), image_size, file);
        }

		fclose(file);
        return _dataset;
    } else {
        printf("Cannot open file!");
		return NULL;
    }
}

uchar* read_mnist_labels(char * full_path, int& number_of_labels) {

    FILE * file = fopen(full_path, "rb");

    if(file != NULL) {
        int magic_number = 0;
        fread((char *)&magic_number, sizeof(magic_number), 1, file);
        magic_number = reverseInt(magic_number);

        if(magic_number != 2049){
			printf("Invalid MNIST label file!");
			return NULL;
		}

        fread((char *)&number_of_labels, sizeof(number_of_labels), 1, file);
		number_of_labels = reverseInt(number_of_labels);

        uchar * _dataset = (uchar *)malloc(sizeof(uchar) * number_of_labels);
        for(int i = 0; i < number_of_labels; i++) {
            fread((char*)&_dataset[i], 1, 1, file);
        }

		fclose(file);
        return _dataset;
    } else {
        printf("Unable to open file!");
		return NULL;
    }
}