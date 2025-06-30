#include <stdlib.h>
#include <stdio.h>
#include "random_forest.h"
#include <time.h>

double gini_index(Sample* left, int n_left, Sample* right, int n_right) {
    int count[2];
    double gini = 0.0;
    int total = n_left + n_right;

    for (int side = 0; side < 2; ++side) {
        Sample* group = side == 0 ? left : right;
        int size = side == 0 ? n_left : n_right;
        count[0] = count[1] = 0;

        for (int i = 0; i < size; ++i)
            count[group[i].label]++;

        double score = 0.0;
        for (int c = 0; c < 2; ++c) {
            double p = (double)count[c] / size;
            score += p * p;
        }

        gini += (1.0 - score) * ((double)size / total);
    }
    return gini;
}

DecisionStump train_stump(Sample* data, int n_samples, int n_features) {
    DecisionStump best;
    double best_gini = 1e9;

    for (int f = 0; f < n_features; ++f) {
        for (int i = 0; i < n_samples; ++i) {
            double threshold = data[i].features[f];
            Sample left[MAX_SAMPLES], right[MAX_SAMPLES];
            int n_left = 0, n_right = 0;

            for (int j = 0; j < n_samples; ++j) {
                if (data[j].features[f] < threshold)
                    left[n_left++] = data[j];
                else
                    right[n_right++] = data[j];
            }

            double gini = gini_index(left, n_left, right, n_right);
            if (gini < best_gini) {
                best_gini = gini;
                best.feature_index = f;
                best.threshold = threshold;

                int counts[2] = {0};
                for (int j = 0; j < n_left; ++j) counts[left[j].label]++;
                best.left_label = counts[0] > counts[1] ? 0 : 1;

                counts[0] = counts[1] = 0;
                for (int j = 0; j < n_right; ++j) counts[right[j].label]++;
                best.right_label = counts[0] > counts[1] ? 0 : 1;
            }
        }
    }
    return best;
}

void bootstrap(Sample* data, int n_samples, Sample* out_sample) {
    for (int i = 0; i < n_samples; ++i) {
        int idx = rand() % n_samples;
        out_sample[i] = data[idx];
    }
}

void train_forest(RandomForest* forest, Sample* data, int n_samples, int n_features, int n_trees) {
    forest->n_trees = n_trees;
    for (int i = 0; i < n_trees; ++i) {
        Sample bootstrap_sample[MAX_SAMPLES];
        bootstrap(data, n_samples, bootstrap_sample);
        forest->trees[i] = train_stump(bootstrap_sample, n_samples, n_features);
    }
}

int predict_tree(DecisionStump* tree, Sample* sample) {
    return sample->features[tree->feature_index] < tree->threshold ? tree->left_label : tree->right_label;
}

int predict_forest(RandomForest* forest, Sample* sample) {
    int votes[2] = {0};
    for (int i = 0; i < forest->n_trees; ++i) {
        int prediction = predict_tree(&forest->trees[i], sample);
        votes[prediction]++;
    }
    return votes[0] > votes[1] ? 0 : 1;
}

void evaluate_model(RandomForest* forest, Sample* data, int n_samples, int* TP, int* FP, int* TN, int* FN) {
    *TP = *FP = *TN = *FN = 0;
    for (int i = 0; i < n_samples; ++i) {
        int pred = predict_forest(forest, &data[i]);
        int true_label = data[i].label;

        if (pred == 1 && true_label == 1) (*TP)++;
        else if (pred == 1 && true_label == 0) (*FP)++;
        else if (pred == 0 && true_label == 0) (*TN)++;
        else if (pred == 0 && true_label == 1) (*FN)++;
    }
}