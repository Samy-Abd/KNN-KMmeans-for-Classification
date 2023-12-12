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

public:
	DatasetLoader(std::string folderPath, float split = 0.8f);
private:
	std::map<int,std::vector<DataPoint>> dataset;
};