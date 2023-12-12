#include "KMeansClustering.h"
#include <random>
#include <limits>
#include <assert.h>

KMeansClustering::KMeansClustering(int k, const DatasetLoader& datasetLoader, int maxIterations)
	:
	k(k),
	maxIterations(maxIterations),
	datasetLoader(datasetLoader)
{
	
}

void KMeansClustering::Fit(int seed)
{
	std::default_random_engine rng(seed);
	std::vector<DataPoint> trainingDataCopy = datasetLoader.GetTrainingData();

	std::shuffle(trainingDataCopy.begin(), trainingDataCopy.end(), rng);

	// Initialize centroids with the first k data points
	std::vector<DataPoint> centroids(trainingDataCopy.begin(), trainingDataCopy.begin() + k);
    //TODO : Check if a copy needs to be made.
    //std::vector<DataPoint> centroidsCopy = centroids;

    for (int iteration = 0; iteration < maxIterations; ++iteration) {
        // Assign each data point to the nearest centroid
        for (DataPoint& point : trainingDataCopy) {
            float minDistance = std::numeric_limits<float>::max();
            int nearestCentroidIndex = -1;

            for (size_t i = 0; i < centroids.size(); ++i) {
                float distance = EucledianDistance(point.data, centroids[i].data);
                if (distance < minDistance) {
                    minDistance = distance;
                    nearestCentroidIndex = int(i);
                }
            }

            point.classIndex = nearestCentroidIndex + 1;
        }

        // Update centroids based on the mean of assigned data points
        for (size_t i = 1; i < centroids.size() + 1; ++i) {
            std::vector<float> sum(centroids[i-1].data.size(), 0.0);
            int count = 0;

            for (const DataPoint& point : trainingDataCopy) {
                if (point.classIndex == static_cast<int>(i)) {
                    for (size_t j = 0; j < sum.size(); ++j) {
                        sum[j] += point.data[j];
                    }
                    ++count;
                }
            }

            if (count > 0) {
                for (size_t j = 0; j < sum.size(); ++j) {
                    centroids[i - 1].data[j] = sum[j] / count;
                }
            }
        }
    }

    this->centroids = centroids;

}

void KMeansClustering::Fit()
{
	Fit(std::random_device()());
}

int KMeansClustering::PredictOne(const DataPoint& dataPoint) const
{
    assert(this->centroids.size() > 0); //If assertion fails : Fit() was not called before trying to predict.
    float minDistance = std::numeric_limits<float>::max();
    int predictedClass = -1;

    for (size_t i = 0; i < centroids.size(); ++i) {
        float distance = EucledianDistance(dataPoint.data, centroids[i].data);
        if (distance < minDistance) {
            minDistance = distance;
            predictedClass = int(i);
        }
    }

    return predictedClass;
}

std::vector<int> KMeansClustering::Predict(std::vector<DataPoint> queryList) const
{
    std::vector<int> results;
    for (const DataPoint& query : queryList)
    {
        results.push_back(PredictOne(query));
    }
    return results;
}

float KMeansClustering::EucledianDistance(const std::vector<float>& point1, const std::vector<float>& point2) const
{
    float distance = 0.0;
    for (size_t i = 0; i < point1.size(); ++i) {
        distance += powf(point1[i] - point2[i], 2);
    }
    return sqrtf(distance);
}
