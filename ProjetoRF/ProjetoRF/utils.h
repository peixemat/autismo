#ifndef UTILS_H
#define UTILS_H

#include "random_forest.h"

void load_dataset(const char *filename, Sample *data, int *n_samples, int *n_features);
void split_data_train_test(Sample *data, int n_samples, Sample *train_data, int *n_train, Sample *test_data, int *n_test);
void evaluate_model(RandomForest *rf, Sample *data, int n_samples);

#endif