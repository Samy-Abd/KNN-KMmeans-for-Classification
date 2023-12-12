#include "DatasetLoader.h"
#include "KNNAlgorithm.h"
#include <filesystem>

int main()
{
	std::filesystem::path currentPath = std::filesystem::current_path();
	DatasetLoader datasetLoader{ currentPath.string() + "\\..\\images\\E34", 5};
	KNNAlgorithm knn(datasetLoader);
	int result = knn.PredictOne(10, datasetLoader.GetEvaluationData()[9]);
	std::vector<int> results = knn.Predict(10, datasetLoader.GetEvaluationData());
	return 0;
}