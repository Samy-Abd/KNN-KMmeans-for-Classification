#pragma once
#include "DatasetLoader.h"
#include <vector>

class KNNAlgorithm
{
public:
	KNNAlgorithm(const DatasetLoader& datasetLoader);
	int PredictOne(int k, const DataPoint& queryData);
	std::vector<int> Predict(int k, std::vector<DataPoint> queryList);
private:
	float EucledianDistance(const std::vector<float>& point1, const std::vector<float>& point2);
	std::vector<int> FindNeighbors(int k, std::vector<float> queryData);
	int MajorityVote(std::vector<int> neighborsClasses);
private:
	const DatasetLoader& datasetLoader;
};
