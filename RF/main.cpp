#include "DatasetLoader.h"
#include "KNNAlgorithm.h"
#include <filesystem>
#include <iostream>
#include "KMeansClustering.h"
#include "Metrics.h"
#include "KNNEval.h"
#include "KMeansEval.h"
#include <random>

int main()
{
	const std::string REPRESENTATION = "SA";
	std::filesystem::path currentPath = std::filesystem::current_path();
	DatasetLoader datasetLoader{ currentPath.string() + "\\..\\images\\" + REPRESENTATION, 7};
	//KNNAlgorithm knn(datasetLoader);
	//int result = knn.PredictOne(10, datasetLoader.GetEvaluationData()[9]);
	//std::vector<int> results = knn.Predict(10, datasetLoader.GetEvaluationData());

	//KNNEval knnEval = KNNEval(datasetLoader);
	//Metrics knnMetrics = knnEval.Evaluate(3);
	//PrintConfusionMatrix(knnMetrics.confusionMatrix);
	//PrintMetrics(knnMetrics);

	//std::cout << "\n\n------------Kmeans metrics----------\n\n";
	//KMeansClustering kMeans(9, datasetLoader,1000);
	//float time = kMeans.Fit();
	//std::cout<< "Training time : " << time << " ms\n";

	//KMeansEval kMeansEval(kMeans, datasetLoader);
	//Metrics kmeansMetrics = kMeansEval.Evaluate();
	//PrintConfusionMatrix(kmeansMetrics.confusionMatrix);
	//PrintMetrics(kmeansMetrics);

	//Do the kmeans fitting multiple times with different seeds and measure the time.
	constexpr int KMEANS_EVALUATION_ITERATIONS = 10;
	constexpr int KMEANS_MAX_ITERATIONS = 100;
	constexpr int KMEANS_K = 9;
	constexpr int KNN_K = 9;

	std::vector<Metrics> kmeansMetrics;
	std::vector<float> kMeansFitTime;
	std::vector<Metrics> knnMetrics;
	int knnKList[] = {1,3,5,15,datasetLoader.GetTrainingData().size()};

	for (int i = 0; i < KMEANS_EVALUATION_ITERATIONS; ++i)
	{
		KMeansClustering kmeans(KMEANS_K, datasetLoader, KMEANS_MAX_ITERATIONS);
		float time = kmeans.Fit();
		KMeansEval kMeansEval(kmeans, datasetLoader);
		kmeansMetrics.push_back(kMeansEval.Evaluate());
		kMeansFitTime.push_back(time);
	}

	//Generate 100 ints randomly.
	std::mt19937 mt(10);
	std::vector<int> seedListKnn;
	for (int i = 0; i < 100; ++i)
	{
		seedListKnn.push_back(mt());
	}
	std::vector<Metrics> knnMeansByK;
	//Do the knn predictions with different k values and measure the time.
	for (int k : knnKList)
	{
		std::vector<Metrics> metricsForK;
		for (int seed : seedListKnn)
		{
			DatasetLoader datasetLoader{ currentPath.string() + "\\..\\images\\" + REPRESENTATION, seed};
			KNNEval knnEval(datasetLoader);
			metricsForK.push_back(knnEval.Evaluate(k));
		}
		Metrics meanKnnMetrics;
		//Initialize knn metrics to 0
		meanKnnMetrics.accuracy = 0;
		meanKnnMetrics.time = 0;
		meanKnnMetrics.classesPrecisionRecallF1.resize(datasetLoader.GetClassCount());

		for (PrecisionRecallF1& precisionRecallF1 : meanKnnMetrics.classesPrecisionRecallF1)
		{
			precisionRecallF1.f1Score = 0;
			precisionRecallF1.precision = 0;
			precisionRecallF1.recall = 0;
		}
		for (const Metrics& metrics : metricsForK)
		{
			meanKnnMetrics.accuracy += metrics.accuracy;
			meanKnnMetrics.time += metrics.time;
			for (int i = 0; i < meanKnnMetrics.classesPrecisionRecallF1.size(); ++i)
			{
				meanKnnMetrics.classesPrecisionRecallF1[i].precision += metrics.classesPrecisionRecallF1[i].precision;
				meanKnnMetrics.classesPrecisionRecallF1[i].recall += metrics.classesPrecisionRecallF1[i].recall;
				meanKnnMetrics.classesPrecisionRecallF1[i].f1Score += metrics.classesPrecisionRecallF1[i].f1Score;
			}
		}
		//Divide the values for knn
		int divider = metricsForK.size();
		meanKnnMetrics.accuracy = meanKnnMetrics.accuracy / divider;
		meanKnnMetrics.time = meanKnnMetrics.time / divider;
		for (PrecisionRecallF1& precisionRecallF1 : meanKnnMetrics.classesPrecisionRecallF1)
		{
			precisionRecallF1.f1Score = precisionRecallF1.f1Score / divider;
			precisionRecallF1.precision = precisionRecallF1.f1Score / divider;
			precisionRecallF1.recall = precisionRecallF1.f1Score / divider;
		}
		knnMeansByK.push_back(meanKnnMetrics);
	}

	for (int i = 0; i < knnMeansByK.size(); ++i)
	{
		Metrics metrics = knnMeansByK[i];
		std::cout << "\n\nMean KNN metrics for k = " << knnKList[i] << " :\n";
		std::cout << "Confusion matrix :\n";
		PrintConfusionMatrix(metrics.confusionMatrix);
		std::cout << "Accuracy : " << metrics.accuracy << "%; " << "time : " << metrics.time * 1000 << " ms\n";
		for (int j = 0; j < metrics.classesPrecisionRecallF1.size(); ++j)
		{
			std::cout << "class " << j + 1 << "\n";
			std::cout << "Precision : " << metrics.classesPrecisionRecallF1[j].precision * 100 << "%\t";
			std::cout << "Recall : " << metrics.classesPrecisionRecallF1[j].recall * 100 << "%\t";
			std::cout << "f1 score : " << metrics.classesPrecisionRecallF1[j].f1Score * 100 << "%\t";
			std::cout << "\n";
		}
	}
	// Calculate the mean for knn and kmeans
	Metrics meanKnnMetrics;
	Metrics meanKmeansMetrics;
	//Initialize knn metrics to 0
	meanKnnMetrics.accuracy = 0;
	meanKnnMetrics.time = 0;
	meanKnnMetrics.classesPrecisionRecallF1.resize(datasetLoader.GetClassCount());

	for (PrecisionRecallF1& precisionRecallF1 : meanKnnMetrics.classesPrecisionRecallF1)
	{
		precisionRecallF1.f1Score = 0;
		precisionRecallF1.precision = 0;
		precisionRecallF1.recall = 0;
	}

	//Initialize kmeans metrics to 0
	meanKmeansMetrics.accuracy = 0;
	meanKmeansMetrics.time = 0;
	meanKmeansMetrics.classesPrecisionRecallF1.resize(datasetLoader.GetClassCount());
	for (PrecisionRecallF1& precisionRecallF1 : meanKmeansMetrics.classesPrecisionRecallF1)
	{
		precisionRecallF1.f1Score = 0;
		precisionRecallF1.precision = 0;
		precisionRecallF1.recall = 0;
	}


	//Add all of the values for knn
	for (const Metrics& metrics : knnMetrics)
	{
		meanKnnMetrics.accuracy += metrics.accuracy;
		meanKnnMetrics.time += metrics.time;
		for (int i = 0; i < meanKnnMetrics.classesPrecisionRecallF1.size(); ++i)
		{
			meanKnnMetrics.classesPrecisionRecallF1[i].precision += metrics.classesPrecisionRecallF1[i].precision;
			meanKnnMetrics.classesPrecisionRecallF1[i].recall += metrics.classesPrecisionRecallF1[i].recall;
			meanKnnMetrics.classesPrecisionRecallF1[i].f1Score += metrics.classesPrecisionRecallF1[i].f1Score;
		}
	}

	//Add all of the values for kmeans
	for (const Metrics& metrics : kmeansMetrics)
	{
		meanKmeansMetrics.accuracy += metrics.accuracy;
		meanKmeansMetrics.time += metrics.time;
		for (int i = 0; i < meanKmeansMetrics.classesPrecisionRecallF1.size(); ++i)
		{
			meanKmeansMetrics.classesPrecisionRecallF1[i].precision += metrics.classesPrecisionRecallF1[i].precision;
			meanKmeansMetrics.classesPrecisionRecallF1[i].recall += metrics.classesPrecisionRecallF1[i].recall;
			meanKmeansMetrics.classesPrecisionRecallF1[i].f1Score += metrics.classesPrecisionRecallF1[i].f1Score;
		}
	}
	//Divide the values for knn
	int divider = knnMetrics.size();
	meanKnnMetrics.accuracy = meanKnnMetrics.accuracy / divider;
	meanKnnMetrics.time = meanKnnMetrics.time / divider;
	for (PrecisionRecallF1& precisionRecallF1 : meanKnnMetrics.classesPrecisionRecallF1)
	{
		precisionRecallF1.f1Score = precisionRecallF1.f1Score / divider;
		precisionRecallF1.precision = precisionRecallF1.f1Score / divider;
		precisionRecallF1.recall = precisionRecallF1.f1Score / divider;
	}

	//Divide the values for kmeans
	 divider = kmeansMetrics.size();
	meanKmeansMetrics.accuracy = meanKmeansMetrics.accuracy / divider;
	meanKmeansMetrics.time = meanKmeansMetrics.time / divider;
	for (PrecisionRecallF1& precisionRecallF1 : meanKmeansMetrics.classesPrecisionRecallF1)
	{
		precisionRecallF1.f1Score = precisionRecallF1.f1Score / divider;
		precisionRecallF1.precision = precisionRecallF1.f1Score / divider;
		precisionRecallF1.recall = precisionRecallF1.f1Score / divider;
	}
	
	float meanFitTime = 0;
	for (float time : kMeansFitTime)
	{
		meanFitTime += time;
	}
	meanFitTime /= kMeansFitTime.size();

	//Print everything

	//for (int i = 0; i < knnMetrics.size(); ++i)
	//{
	//	Metrics metrics = knnMetrics[i];
	//	std::cout << "\n\nKNN metrics for k = " << knnKList[i] << " :\n";
	//	std::cout << "Confusion matrix :\n";
	//	PrintConfusionMatrix(metrics.confusionMatrix);
	//	std::cout << "Accuracy : " << metrics.accuracy << "%; " << "time : " << metrics.time * 1000 << " ms\n";
	//	for (int j = 0; j < metrics.classesPrecisionRecallF1.size(); ++j)
	//	{
	//		std::cout << "class " << j + 1 << "\n";
	//		std::cout << "Precision : " << metrics.classesPrecisionRecallF1[j].precision * 100 << "%\t";
	//		std::cout << "Recall : " << metrics.classesPrecisionRecallF1[j].recall * 100 << "%\t";
	//		std::cout << "f1 score : " << metrics.classesPrecisionRecallF1[j].f1Score * 100 << "%\t";
	//		std::cout << "\n";
	//	}
	//}
	return 0;
}