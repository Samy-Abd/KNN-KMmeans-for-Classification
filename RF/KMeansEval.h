#pragma once
#include "DatasetLoader.h"
#include "KMeansClustering.h"

struct DataPointInCluster
{
	DataPoint datapoint;
	int cluster;
};


class KMeansEval
{
public:
	KMeansEval(int k, const DatasetLoader& datasetLoader,int maxIterations, int seed);
public:
	void Evaluate();
private:
	KMeansClustering kMeans;
	int k;
	const DatasetLoader& datasetLoader;
};
