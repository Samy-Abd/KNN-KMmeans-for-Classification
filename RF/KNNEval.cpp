#include "KNNEval.h"
#include <random>
#include <iostream>
#include "Timer.h"


KNNEval::KNNEval(const DatasetLoader& datasetLoader)
    :
    datasetLoader(datasetLoader),
    knn(datasetLoader)
{
}

Metrics KNNEval::Evaluate(int k)
{
	const std::vector<DataPoint>& evaluationData = datasetLoader.GetEvaluationData();
    Metrics metrics;
    //Measure the time it takes to predict
    Timer timer;
	std::vector<int> results = knn.Predict(k, evaluationData);
    metrics.time = timer.Mark();
    metrics.accuracy = calculateAccuracy(results, evaluationData);
    metrics.confusionMatrix = calculateConfusionMatrix(results, evaluationData, datasetLoader.GetClassCount());
   for (int classIndex = 0; classIndex < datasetLoader.GetClassCount(); ++classIndex)
    {
        float precision = calculatePrecision(metrics.confusionMatrix, classIndex);
        float recall = calculateRecall(metrics.confusionMatrix, classIndex);
        float f1Score = calculateF1Score(metrics.confusionMatrix, classIndex);
        metrics.classesPrecisionRecallF1.push_back({ precision, recall, f1Score });
    }
    return metrics;
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
