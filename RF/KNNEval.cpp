#include "KNNEval.h"
#include <random>
#include <iostream>

KNNEval::KNNEval(const DatasetLoader& datasetLoader)
    :
    datasetLoader(datasetLoader),
    knn(datasetLoader)
{
}




KNNMetrics KNNEval::Evaluate(int k)
{
	const std::vector<DataPoint>& evaluationData = datasetLoader.GetEvaluationData();
    KNNMetrics metrics;
	std::vector<int> results = knn.Predict(k, evaluationData);
    metrics.accuracy = calculateAccuracy(results, evaluationData);
    ConfusionMatrix confusionMatrix = calculateConfusionMatrix(results, evaluationData, datasetLoader.GetClassCount());
    PrintConfusionMatrix(confusionMatrix);
   for (int classIndex = 0; classIndex < datasetLoader.GetClassCount(); ++classIndex)
    {
        float precision = calculatePrecision(confusionMatrix, classIndex);
        float recall = calculateRecall(confusionMatrix, classIndex);
        float f1Score = calculateF1Score(confusionMatrix, classIndex);
        metrics.classesPrecisionRecallF1.push_back({ precision, recall, f1Score });
    }
    return metrics;
}

void KNNEval::PrintConfusionMatrix(const ConfusionMatrix& confusionMatrix)
{
    std::cout << "Predicted label on top, real label on the left\n";
    for (int i = 0; i < confusionMatrix.size(); ++i)
    {
        for (int j = 0; j < confusionMatrix.size(); ++j)
        {
            std::cout << confusionMatrix[i][j] << ' ';
        }
        std::cout << "\n";
    }
 
}

ConfusionMatrix KNNEval::calculateConfusionMatrix(const std::vector<int>&predicted, const std::vector<DataPoint>&actual, int numClasses)
{
    std::vector<std::vector<int>> confusionMatrix(numClasses, std::vector<int>(numClasses, 0));

    for (size_t i = 0; i < predicted.size(); ++i) {
        confusionMatrix[actual[i].classIndex-1][predicted[i]-1]++;
    }

    return confusionMatrix;
}

float KNNEval::calculateAccuracy(const std::vector<int>& predicted, const std::vector<DataPoint>& actual)
{
    size_t correctCount = 0;

    for (size_t i = 0; i < predicted.size(); ++i) {
        if (predicted[i] == actual[i].classIndex) {
            correctCount++;
        }
    }

    return float(correctCount) / predicted.size();
}

float KNNEval::calculatePrecision(const ConfusionMatrix& confusionMatrix, int classIndex)
{
    int truePositive = confusionMatrix[classIndex][classIndex];
    int falsePositive = 0;

    for (size_t i = 0; i < confusionMatrix.size(); ++i) {
        if (i != static_cast<size_t>(classIndex)) {
            falsePositive += confusionMatrix[i][classIndex];
        }
    }

    if (truePositive + falsePositive == 0) {
        return 0.0;
    }

    return float(truePositive) / (truePositive + falsePositive);
}

float KNNEval::calculateRecall(const ConfusionMatrix& confusionMatrix, int classIndex)
{
    int truePositive = confusionMatrix[classIndex][classIndex];
    int falseNegative = 0;

    for (size_t i = 0; i < confusionMatrix.size(); ++i) {
        if (i != static_cast<size_t>(classIndex)) {
            falseNegative += confusionMatrix[classIndex][i];
        }
    }

    if (truePositive + falseNegative == 0) {
        return 0.0;
    }

    return float(truePositive) / (truePositive + falseNegative);
}

float KNNEval::calculateF1Score(const ConfusionMatrix& confusionMatrix, int classIndex)
{
    double precision = calculatePrecision(confusionMatrix, classIndex);
    double recall = calculateRecall(confusionMatrix, classIndex);

    if (precision + recall == 0) {
        return 0.0;
    }

    return 2.0 * (precision * recall) / (precision + recall);
}
