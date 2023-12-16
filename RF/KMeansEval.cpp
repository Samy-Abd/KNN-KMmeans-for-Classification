#include "KMeansEval.h"

KMeansEval::KMeansEval(int k, const DatasetLoader& datasetLoader, int maxIterations, int seed)
	:
	kMeans(k, datasetLoader, maxIterations),
	datasetLoader(datasetLoader),
	k(k)
{
	kMeans.Fit(seed);
}

void KMeansEval::Evaluate()
{
	//Do the prediction on the evaluation data.
	std::vector<DataPoint> evaluationData = datasetLoader.GetEvaluationData();
	std::vector<int> predictions = kMeans.Predict(evaluationData);
	std::vector<DataPointInCluster> dataPointsInClusters;
	//Add the cluster info to the datapoints
	for (int i = 0; i < predictions.size(); ++i)
	{
		dataPointsInClusters.push_back({evaluationData[i], predictions[i]});
	}

	//Now that we know the real class and the predicted cluster of every element. divide them by cluster.
  	//For every cluster, the class with the most elements takes over.

	//Concatenate all of the elements into one vector

	//Calculate the metrics.
}
