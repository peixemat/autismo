#include <stdio.h>
#include "random_forest.h"
#include "utils.h"

int main() {
    Sample data[MAX_DATA];
    int n_samples, n_features;

    load_dataset("data/darwin.data", data, &n_samples, &n_features);

    Sample train_data[MAX_DATA], test_data[MAX_DATA];
    int n_train, n_test;

    split_data_train_test(data, n_samples, train_data, &n_train, test_data, &n_test);

    RandomForest rf;
    train_random_forest(&rf, train_data, n_train, n_features, 10, 5);

    printf("\nAvaliação Treinamento:\n");
    evaluate_model(&rf, train_data, n_train);

    printf("\nAvaliação Teste:\n");
    evaluate_model(&rf, test_data, n_test);

    for (int i = 0; i < rf.num_trees; i++) {
        free_tree(rf.trees[i].root);
    }
    return 0;
}