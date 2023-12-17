#include "KMeansEval.h"
#include <vector>
#include <unordered_map>
#include <algorithm>

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



KMeansEval::KMeansEval(int k, const DatasetLoader& datasetLoader, int maxIterations, int seed)
	:
	kMeans(k, datasetLoader, maxIterations),
	datasetLoader(datasetLoader),
	k(k)
{
	kMeans.Fit(seed);
}

void KMeansEval::Evaluate()
{
	//Do the prediction on the evaluation data.
	std::vector<DataPoint> evaluationData = datasetLoader.GetEvaluationData();
	std::vector<int> predictions = kMeans.Predict(evaluationData);

    auto test = GetClusterInfo(kMeans, datasetLoader.GetTrainingData());

	//Now that we know the real class and the predicted cluster of every element. divide them by cluster.
  	//For every cluster, the class with the most elements takes over.

	//Concatenate all of the elements into one vector

	//Calculate the metrics.
}
