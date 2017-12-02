#ifndef READ
#define READ
#include "structures.h"

uchar* read_mnist_images(char * full_path, int& number_of_images, int& image_size);
uchar* read_mnist_labels(char * full_path, int& number_of_labels);

#endif