#pragma once
#include <vector>
#include "DatasetLoader.h"
#include "KNNAlgorithm.h"

using ConfusionMatrix = std::vector<std::vector<int>>;

struct PrecisionRecallF1
{
	float precision;
	float recall;
	float f1Score;
};
struct KNNMetrics
{
	float accuracy;
	std::vector<PrecisionRecallF1> classesPrecisionRecallF1;
};


class KNNEval
{
public:
	KNNEval(const DatasetLoader& datasetLoader);
public:
	KNNMetrics Evaluate(int k);
	static void PrintConfusionMatrix(const ConfusionMatrix& confusionMatrix);
private:
    ConfusionMatrix calculateConfusionMatrix(const std::vector<int>& predicted, const std::vector<DataPoint>& actual, int numClasses);
    float calculateAccuracy(const std::vector<int>& predicted, const std::vector<DataPoint>& actual);
    float calculatePrecision(const ConfusionMatrix& confusionMatrix, int classIndex);
    float calculateRecall(const ConfusionMatrix& confusionMatrix, int classIndex);
    float calculateF1Score(const ConfusionMatrix& confusionMatrix, int classIndex);
private:
	KNNAlgorithm knn;
	const DatasetLoader& datasetLoader;
	int seed;
	std::vector<int> kList;
};