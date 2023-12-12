#include "KNNAlgorithm.h"
#include <algorithm>

KNNAlgorithm::KNNAlgorithm(const DatasetLoader& datasetLoader)
	:
	datasetLoader(datasetLoader)
{
}

int KNNAlgorithm::PredictOne(int k, const DataPoint& queryData)
{
    std::vector<int> neighbors = FindNeighbors(k, queryData.data);
    return MajorityVote(neighbors);
}

std::vector<int> KNNAlgorithm::Predict(int k, std::vector<DataPoint> queryList)
{
    std::vector<int> results;
    for (const DataPoint& query : queryList)
    {
        results.push_back(PredictOne(k, query));
    }
    return results;
}

float KNNAlgorithm::EucledianDistance(const std::vector<float>& point1, const std::vector<float>& point2)
{
    float distance = 0.0;
    for (size_t i = 0; i < point1.size(); ++i) {
        distance += powf(point1[i] - point2[i], 2);
    }
    return sqrtf(distance);
}

std::vector<int> KNNAlgorithm::FindNeighbors(int k, std::vector<float> queryData)
{
    std::vector<std::pair<float, int>> distances; // distance, class

    for (const auto& [classIndex, data] : datasetLoader.GetTrainingData()) {
        float distance = EucledianDistance(queryData, data);
        distances.push_back({ distance, classIndex });
    }

    // Sort distances in ascending order
    std::sort(distances.begin(), distances.end());

    // Return the classes of the first k neighbors
    std::vector<int> neighborsClasses;
    for (int i = 0; i < k; ++i) {
        neighborsClasses.push_back(distances[i].second);
    }

    return neighborsClasses;
}

int KNNAlgorithm::MajorityVote(std::vector<int> neighborsClasses)
{
    // Count occurrences of each class
    std::vector<int> classCount(*std::max_element(neighborsClasses.begin(), neighborsClasses.end()) + 1, 0);

    for (int neighborClass : neighborsClasses) {
        classCount[neighborClass]++;
    }

    // Find the class with the maximum count
    int maxCount = -1;
    int predictedClass = -1;

    for (size_t i = 0; i < classCount.size(); ++i) {
        if (classCount[i] > maxCount) {
            maxCount = classCount[i];
            predictedClass = static_cast<int>(i);
        }
    }

    return predictedClass;
}
