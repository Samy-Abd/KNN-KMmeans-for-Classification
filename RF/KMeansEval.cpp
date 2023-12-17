#include "KMeansEval.h"
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <assert.h>

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

void KMeansEval::Evaluate()
{

	//Now that we know the real class and the predicted cluster of every element. divide them by cluster.
  	//For every cluster, the class with the most elements takes over.
    std::vector<int> predictedClusters = kMeans.Predict(datasetLoader.GetEvaluationData());
    
    // Map the clusters to their majority classes
    std::vector<int>  predictedClasses = Predict(datasetLoader.GetEvaluationData());
	//Calculate the metrics.
}
