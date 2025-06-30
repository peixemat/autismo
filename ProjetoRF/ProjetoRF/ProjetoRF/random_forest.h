#ifndef RANDOM_FOREST_H
#define RANDOM_FOREST_H
#include "utils.h"
#define MAX_FEATURES 100
#define MAX_TREES 100
#define MAX_DEPTH 10
#define MAX_DATA 1000

typedef struct {
    double features[MAX_FEATURES];
    int label;
} Sample;

typedef struct Node {
    int feature_index;
    double threshold;
    int prediction;
    struct Node *left;
    struct Node *right;
} Node;

typedef struct {
    Node *root;
} DecisionTree;

typedef struct {
    DecisionTree trees[MAX_TREES];
    int num_trees;
    int max_depth;
    int num_features;
} RandomForest;

void train_random_forest(RandomForest *rf, Sample *data, int n_samples, int n_features, int n_trees, int max_depth);
int predict_random_forest(RandomForest *rf, double *features);
void free_tree(Node *node);

#endif