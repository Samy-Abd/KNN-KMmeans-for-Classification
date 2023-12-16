#include "DatasetLoader.h"
#include "KNNAlgorithm.h"
#include <filesystem>
#include "KMeansClustering.h"
#include "Metrics.h"
#include "KNNEval.h"

int main()
{
	std::filesystem::path currentPath = std::filesystem::current_path();
	DatasetLoader datasetLoader{ currentPath.string() + "\\..\\images\\E34", 5};
	KNNAlgorithm knn(datasetLoader);
	int result = knn.PredictOne(10, datasetLoader.GetEvaluationData()[9]);
	std::vector<int> results = knn.Predict(10, datasetLoader.GetEvaluationData());

	KMeansClustering kMeans(9, datasetLoader);
	kMeans.Fit();
	kMeans.Predict(datasetLoader.GetEvaluationData());

	KNNEval knnEval = KNNEval(datasetLoader);
	KNNMetrics knnMetrics = knnEval.Evaluate(10);
	KNNEval::PrintConfusionMatrix(knnMetrics.confusionMatrix);
	KNNEval::PrintMetrics(knnMetrics);
	return 0;
}