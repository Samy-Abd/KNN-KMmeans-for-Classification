#pragma once
#include "DatasetLoader.h"

class KMeansClustering
{
public:
	KMeansClustering(int k, const DatasetLoader& datasetLoader, int maxIterations = 100);
public:
	void Fit(int seed);
	void Fit();
	int PredictOne(const DataPoint& dataPoint) const;
	std::vector<int> Predict(std::vector<DataPoint> queryList) const;
private: 
	float EucledianDistance(const std::vector<float>& point1, const std::vector<float>& point2) const;
private:
	int k;
	int maxIterations;
	const DatasetLoader& datasetLoader;
	std::vector<DataPoint> centroids;
};
