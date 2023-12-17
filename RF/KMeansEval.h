#pragma once
#include "DatasetLoader.h"
#include "KMeansClustering.h"

struct ClusterInfo {
	std::vector<float> centroid;
	int majorityClass;
};

class KMeansEval
{
public:
	KMeansEval(const KMeansClustering& kmeans, const DatasetLoader& datasetLoader);
public:
	void Evaluate();
	std::vector<ClusterInfo> GetClusterInfo(const KMeansClustering& kmeans, const std::vector<DataPoint>& dataPoints);
	int PredictOne(const DataPoint& datapoint);
	std::vector<int> Predict(std::vector<DataPoint> datapoints);
private:
	float EucledianDistance(const std::vector<float>& point1, const std::vector<float>& point2) const;
private:
	const KMeansClustering& kMeans;
	const DatasetLoader& datasetLoader;
	std::vector<ClusterInfo> clustersInfo;
};
