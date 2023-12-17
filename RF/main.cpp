#include "DatasetLoader.h"
#include "KNNAlgorithm.h"
#include <filesystem>
#include "KMeansClustering.h"
#include "Metrics.h"
#include "KNNEval.h"
#include "KMeansEval.h"

int main()
{
	std::filesystem::path currentPath = std::filesystem::current_path();
	DatasetLoader datasetLoader{ currentPath.string() + "\\..\\images\\GFD", 5};
	KNNAlgorithm knn(datasetLoader);
	int result = knn.PredictOne(10, datasetLoader.GetEvaluationData()[9]);
	std::vector<int> results = knn.Predict(10, datasetLoader.GetEvaluationData());

	KNNEval knnEval = KNNEval(datasetLoader);
	KNNMetrics knnMetrics = knnEval.Evaluate(3);
	KNNEval::PrintConfusionMatrix(knnMetrics.confusionMatrix);
	KNNEval::PrintMetrics(knnMetrics);



	KMeansClustering kMeans(9, datasetLoader);
	kMeans.Fit(2);
	auto kek = kMeans.Predict(datasetLoader.GetEvaluationData());

	KMeansEval kMeansEval(kMeans, datasetLoader);
	kMeansEval.Evaluate();
	return 0;
}