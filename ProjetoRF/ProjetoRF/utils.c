#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "utils.h"

void load_dataset(const char *filename, Sample *data, int *n_samples, int *n_features) {
    FILE *file = fopen(filename, "r");
    if (!file) {
        printf("Erro abrindo arquivo\n");
        exit(1);
    }
    char line[1024];
    int idx = 0;
    while (fgets(line, sizeof(line), file)) {
        char *token = strtok(line, ",");
        int f = 0;
        while (token && f < MAX_FEATURES) {
            if (strchr(token, '\n')) token[strlen(token) - 1] = '\0';
            if (token[0] == '0' || token[0] == '1') {
                data[idx].label = atoi(token);
            } else {
                data[idx].features[f++] = atof(token);
            }
            token = strtok(NULL, ",");
        }
        idx++;
    }
    *n_samples = idx;
    *n_features = f;
    fclose(file);
}

void split_data_train_test(Sample *data, int n_samples, Sample *train_data, int *n_train, Sample *test_data, int *n_test) {
    int train_size = (int)(n_samples * 0.8);
    for (int i = 0; i < train_size; i++) {
        train_data[i] = data[i];
    }
    for (int i = train_size; i < n_samples; i++) {
        test_data[i - train_size] = data[i];
    }
    *n_train = train_size;
    *n_test = n_samples - train_size;
}

void evaluate_model(RandomForest *rf, Sample *data, int n_samples) {
    int TP = 0, TN = 0, FP = 0, FN = 0;
    for (int i = 0; i < n_samples; i++) {
        int pred = predict_random_forest(rf, data[i].features);
        if (pred == 1 && data[i].label == 1) TP++;
        if (pred == 0 && data[i].label == 0) TN++;
        if (pred == 1 && data[i].label == 0) FP++;
        if (pred == 0 && data[i].label == 1) FN++;
    }

    double err = (double)(FP + FN) / (TP + TN + FP + FN);
    double acc = 1 - err;
    double pre = (TP + FP) ? (double)TP / (TP + FP) : 0;
    double rec = (TP + FN) ? (double)TP / (TP + FN) : 0;
    double f1 = (pre + rec) ? (2 * pre * rec) / (pre + rec) : 0;

    printf("Confusion Matrix:\n");
    printf("TP = %d\n", TP);
    printf("TN = %d\n", TN);
    printf("FP = %d\n", FP);
    printf("FN = %d\n", FN);
    printf("Err = %.4f\n", err);
    printf("Acc = %.4f\n", acc);
    printf("Pre = %.4f\n", pre);
    printf("Rec = %.4f\n", rec);
    printf("F1 = %.4f\n", f1);
}