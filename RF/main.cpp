#include "DatasetLoader.h"
#include "KNNAlgorithm.h"
#include <filesystem>
#include <iostream>
#include "KMeansClustering.h"
#include "Metrics.h"
#include "KNNEval.h"
#include "KMeansEval.h"

int main()
{
	std::filesystem::path currentPath = std::filesystem::current_path();
	DatasetLoader datasetLoader{ currentPath.string() + "\\..\\images\\SA", 5};
	KNNAlgorithm knn(datasetLoader);
	int result = knn.PredictOne(10, datasetLoader.GetEvaluationData()[9]);
	std::vector<int> results = knn.Predict(10, datasetLoader.GetEvaluationData());

	KNNEval knnEval = KNNEval(datasetLoader);
	Metrics knnMetrics = knnEval.Evaluate(3);
	PrintConfusionMatrix(knnMetrics.confusionMatrix);
	PrintMetrics(knnMetrics);

	std::cout << "\n\n------------Kmeans metrics----------\n\n";
	KMeansClustering kMeans(9, datasetLoader,1000);
	float time = kMeans.Fit();
	std::cout<< "Training time : " << time << " ms\n";

	KMeansEval kMeansEval(kMeans, datasetLoader);
	Metrics kmeansMetrics = kMeansEval.Evaluate();
	PrintConfusionMatrix(kmeansMetrics.confusionMatrix);
	PrintMetrics(kmeansMetrics);
	return 0;
}