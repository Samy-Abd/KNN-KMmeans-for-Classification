#pragma once
#include <string>
#include <vector>
#include <map>

//Represents a datapoint along with the class it belongs to.
struct DataPoint {
	DataPoint(int classIndex, std::vector<float> data);
	int classIndex;
	std::vector<float> data;
};

class DatasetLoader
{
public:
	DatasetLoader(std::string folderPath, int seed, float trainingRatio = 0.8f);
	DatasetLoader(std::string folderPath, float trainingRatio = 0.8f);
public:
	const std::vector<DataPoint>& GetTrainingData() const;
	const std::vector<DataPoint>& GetEvaluationData() const;
private:
	std::map<int,std::vector<DataPoint>> dataset;
	std::vector<DataPoint> trainingData;
	std::vector<DataPoint> evaluationData;
};