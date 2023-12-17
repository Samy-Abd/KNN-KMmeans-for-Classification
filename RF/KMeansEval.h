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
	KMeansEval(int k, const DatasetLoader& datasetLoader,int maxIterations, int seed);
public:
	void Evaluate();
	std::vector<ClusterInfo> GetClusterInfo(const KMeansClustering& kmeans, const std::vector<DataPoint>& dataPoints);
private:
	KMeansClustering kMeans;
	int k;
	const DatasetLoader& datasetLoader;
};
