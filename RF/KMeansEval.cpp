#include "KMeansEval.h"
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <assert.h>
#include "Timer.h"

std::vector<ClusterInfo> KMeansEval::GetClusterInfo(const KMeansClustering& kmeans, const std::vector<DataPoint>& dataPoints) {
    std::vector<ClusterInfo> clusterInfos(kmeans.GetK());
    std::vector<std::unordered_map<int, int>> classCounts(kmeans.GetK());
    
    // Count class occurrences in each cluster
    for (const auto& point : dataPoints) {
        int clusterIndex = kmeans.PredictOne(point);
        classCounts[clusterIndex][point.classIndex]++;
    }

    // Determine majority class for each cluster
    for (size_t i = 0; i < kmeans.GetK(); ++i) {
        int maxCount = 0;
        int majorityClass = -1;

        for (const auto& classCount : classCounts[i]) {
            if (classCount.second > maxCount) {
                maxCount = classCount.second;
                majorityClass = classCount.first;
            }
        }
        
        clusterInfos[i].centroid = kmeans.GetCentroids()[i].data;
        clusterInfos[i].majorityClass = majorityClass;
    }

    return clusterInfos;
}

int KMeansEval::PredictOne(const DataPoint& datapoint)
{
    assert(this->clustersInfo.size() > 0); //If assertion fails : Fit() was not called before trying to predict.
    float minDistance = std::numeric_limits<float>::max();
    int predictedClass = -1;

    for (size_t i = 0; i < clustersInfo.size(); ++i) {
        float distance = EucledianDistance(datapoint.data, clustersInfo[i].centroid);
        if (distance < minDistance) {
            minDistance = distance;
            predictedClass = clustersInfo[i].majorityClass;
        }
    }

    return predictedClass;
}

std::vector<int> KMeansEval::Predict(std::vector<DataPoint> datapoints)
{
    std::vector<int> results;
    results.reserve(datapoints.size());
    for (const DataPoint& query : datapoints)
    {
        results.push_back(PredictOne(query));
    }
    return results;
}

float KMeansEval::EucledianDistance(const std::vector<float>& point1, const std::vector<float>& point2) const
{
    float distance = 0.0;
    for (size_t i = 0; i < point1.size(); ++i) {
        distance += powf(point1[i] - point2[i], 2);
    }
    return sqrtf(distance);
}



KMeansEval::KMeansEval(const KMeansClustering& kmeans, const DatasetLoader& datasetLoader)
	:
    kMeans(kmeans),
	datasetLoader(datasetLoader),
    clustersInfo(GetClusterInfo(kMeans, datasetLoader.GetTrainingData()))
{
}

Metrics KMeansEval::Evaluate()
{
    const std::vector<DataPoint>& evaluationData = datasetLoader.GetEvaluationData();
    // Map the clusters to their majority classes
    Timer timer;
    Metrics metrics;
    std::vector<int>  results= Predict(evaluationData);
    metrics.time = timer.Mark();
	//Calculate the metrics.
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



ConfusionMatrix KMeansEval::calculateConfusionMatrix(const std::vector<int>& predicted, const std::vector<DataPoint>& actual, int numClasses)
{
    std::vector<std::vector<int>> confusionMatrix(numClasses, std::vector<int>(numClasses, 0));

    for (size_t i = 0; i < predicted.size(); ++i) {
        confusionMatrix[actual[i].classIndex - 1][predicted[i] - 1]++;
    }

    return confusionMatrix;
}

float KMeansEval::calculateAccuracy(const std::vector<int>& predicted, const std::vector<DataPoint>& actual)
{
    size_t correctCount = 0;

    for (size_t i = 0; i < predicted.size(); ++i) {
        if (predicted[i] == actual[i].classIndex) {
            correctCount++;
        }
    }

    return float(correctCount) / predicted.size();
}

float KMeansEval::calculatePrecision(const ConfusionMatrix& confusionMatrix, int classIndex)
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

float KMeansEval::calculateRecall(const ConfusionMatrix& confusionMatrix, int classIndex)
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

float KMeansEval::calculateF1Score(const ConfusionMatrix& confusionMatrix, int classIndex)
{
    double precision = calculatePrecision(confusionMatrix, classIndex);
    double recall = calculateRecall(confusionMatrix, classIndex);

    if (precision + recall == 0) {
        return 0.0;
    }

    return 2.0 * (precision * recall) / (precision + recall);
}
