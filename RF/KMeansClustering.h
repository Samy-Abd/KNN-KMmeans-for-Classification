#pragma once
#include "DatasetLoader.h"
#include "Metrics.h"

class KMeansClustering
{
public:
	KMeansClustering(int k, const DatasetLoader& datasetLoader, int maxIterations = 100);
public:
	float Fit(int seed);
	float Fit();
	int PredictOne(const DataPoint& dataPoint) const;
	std::vector<int> Predict(std::vector<DataPoint> queryList) const;
	int GetK() const;
	const std::vector<DataPoint>& GetCentroids() const;
private: 
	float EucledianDistance(const std::vector<float>& point1, const std::vector<float>& point2) const;
private:
	int k;
	int maxIterations;
	const DatasetLoader& datasetLoader;
	std::vector<DataPoint> centroids;
};
