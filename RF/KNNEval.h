#pragma once
#include <vector>
#include "DatasetLoader.h"
#include "KNNAlgorithm.h"
#include "Metrics.h"



class KNNEval
{
public:
	KNNEval(const DatasetLoader& datasetLoader);
public:
	Metrics Evaluate(int k);
private:
    ConfusionMatrix calculateConfusionMatrix(const std::vector<int>& predicted, const std::vector<DataPoint>& actual, int numClasses);
    float calculateAccuracy(const std::vector<int>& predicted, const std::vector<DataPoint>& actual);
    float calculatePrecision(const ConfusionMatrix& confusionMatrix, int classIndex);
    float calculateRecall(const ConfusionMatrix& confusionMatrix, int classIndex);
    float calculateF1Score(const ConfusionMatrix& confusionMatrix, int classIndex);
private:
	KNNAlgorithm knn;
	const DatasetLoader& datasetLoader;
};