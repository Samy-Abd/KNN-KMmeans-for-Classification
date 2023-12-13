#pragma once
#include <vector>

namespace metrics {

    using ConfusionMatrix = std::vector<std::vector<int>>;

    ConfusionMatrix create_confusion_matrix(const std::vector<int>& true_labels, const std::vector<int>& predicted_labels, int num_classes);
    float accuracy(const ConfusionMatrix& matrix);
    std::vector<float> precision(const ConfusionMatrix& matrix);
    std::vector<float> recall(const ConfusionMatrix& matrix);
    std::vector<float> f1_score(const std::vector<float>& precisions, const std::vector<float>& recalls);

} // namespace metrics