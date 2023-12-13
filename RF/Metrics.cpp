#include <vector>
#include <algorithm>
#include <numeric>

namespace metrics {

    using ConfusionMatrix = std::vector<std::vector<int>>;

    // Function to create a confusion matrix
    ConfusionMatrix create_confusion_matrix(const std::vector<int>& true_labels, const std::vector<int>& predicted_labels, int num_classes) {
        ConfusionMatrix matrix(num_classes, std::vector<int>(num_classes, 0));
        for (size_t i = 0; i < true_labels.size(); ++i) {
            matrix[true_labels[i]][predicted_labels[i]]++;
        }
        return matrix;
    }

    // Function to calculate accuracy
    float accuracy(const ConfusionMatrix& matrix) {
        int total = 0, correct = 0;
        for (size_t i = 0; i < matrix.size(); ++i) {
            for (size_t j = 0; j < matrix[i].size(); ++j) {
                if (i == j) correct += matrix[i][j];
                total += matrix[i][j];
            }
        }
        return total > 0 ? static_cast<float>(correct) / total : 0;
    }

    // Function to calculate precision for each class
    std::vector<float> precision(const ConfusionMatrix& matrix) {
        std::vector<float> precisions(matrix.size(), 0);
        for (size_t i = 0; i < matrix.size(); ++i) {
            int sum_col = 0;
            for (size_t j = 0; j < matrix.size(); ++j) {
                sum_col += matrix[j][i];
            }
            precisions[i] = sum_col > 0 ? static_cast<float>(matrix[i][i]) / sum_col : 0;
        }
        return precisions;
    }

    // Function to calculate recall for each class
    std::vector<float> recall(const ConfusionMatrix& matrix) {
        std::vector<float> recalls(matrix.size(), 0);
        for (size_t i = 0; i < matrix.size(); ++i) {
            int sum_row = std::accumulate(matrix[i].begin(), matrix[i].end(), 0);
            recalls[i] = sum_row > 0 ? static_cast<float>(matrix[i][i]) / sum_row : 0;
        }
        return recalls;
    }

    // Function to calculate F1 score for each class
    std::vector<float> f1_score(const std::vector<float>& precisions, const std::vector<float>& recalls) {
        std::vector<float> f1_scores(precisions.size(), 0);
        for (size_t i = 0; i < precisions.size(); ++i) {
            f1_scores[i] = (precisions[i] + recalls[i]) > 0 ? 2 * (precisions[i] * recalls[i]) / (precisions[i] + recalls[i]) : 0;
        }
        return f1_scores;
    }

}
