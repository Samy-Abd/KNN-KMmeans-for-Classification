#pragma once
#include "DatasetLoader.h"
#include "KMeansClustering.h"
#include "Metrics.h"

struct ClusterInfo {
	std::vector<float> centroid;
	int majorityClass;
};




class KMeansEval
{
public:
	KMeansEval(const KMeansClustering& kmeans, const DatasetLoader& datasetLoader);
public:
	Metrics Evaluate();
	std::vector<ClusterInfo> GetClusterInfo(const KMeansClustering& kmeans, const std::vector<DataPoint>& dataPoints);
private:
	int PredictOne(const DataPoint& datapoint);
	std::vector<int> Predict(std::vector<DataPoint> datapoints);
	float EucledianDistance(const std::vector<float>& point1, const std::vector<float>& point2) const;
	ConfusionMatrix calculateConfusionMatrix(const std::vector<int>& predicted, const std::vector<DataPoint>& actual, int numClasses);
	float calculateAccuracy(const std::vector<int>& predicted, const std::vector<DataPoint>& actual);
	float calculatePrecision(const ConfusionMatrix& confusionMatrix, int classIndex);
	float calculateRecall(const ConfusionMatrix& confusionMatrix, int classIndex);
	float calculateF1Score(const ConfusionMatrix& confusionMatrix, int classIndex);
private:
	const KMeansClustering& kMeans;
	const DatasetLoader& datasetLoader;
	std::vector<ClusterInfo> clustersInfo;
};
