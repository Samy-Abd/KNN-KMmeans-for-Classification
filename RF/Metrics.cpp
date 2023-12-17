#include "Metrics.h"

void PrintConfusionMatrix(const ConfusionMatrix& confusionMatrix)
{
    for (int i = 0; i < confusionMatrix.size(); ++i)
    {
        for (int j = 0; j < confusionMatrix.size(); ++j)
        {
            std::cout << confusionMatrix[j][i] << ' ';
        }
        std::cout << "\n";
    }

}

void PrintMetrics(const Metrics& metrics)
{
    std::cout << std::fixed << std::cout.precision(2);
    std::cout << "Accuracy : " << metrics.accuracy * 100 << "%\n";
    std::cout << "Prediction time : " << metrics.time * 1000 << " ms\n";
    for (int i = 0; i < metrics.classesPrecisionRecallF1.size(); ++i)
    {
        std::cout << "class " << i + 1 << "\n";
        std::cout << "Precision : " << metrics.classesPrecisionRecallF1[i].precision * 100<< "%\t";
        std::cout << "Recall : " << metrics.classesPrecisionRecallF1[i].recall * 100 << "%\t";
        std::cout << "f1 score : " << metrics.classesPrecisionRecallF1[i].f1Score * 100 << "%\t";
        std::cout << "\n";
    }
}
