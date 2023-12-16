#include "DatasetLoader.h"
#include <filesystem>
#include <iostream>
#include <fstream>
#include <sstream>
#include <random>

DatasetLoader::DatasetLoader(std::string folderPath, int seed, float trainingRatio)
{
    //Load the images and separate them by class
    for (const auto& entry : std::filesystem::directory_iterator(folderPath)) {
        if (entry.is_regular_file()) {
            // Get the file name from the path
            std::string fileName = entry.path().filename().string();

            // Parse file name to get class number
            int classNumber;
            if (sscanf_s(fileName.c_str(), "s%d", &classNumber) == 1 || sscanf_s(fileName.c_str(), "S%d", &classNumber) == 1) {

                std::vector<float> fileData;
                // Read data from the file
                std::ifstream file(entry.path());
                if (file.is_open()) {
                    std::string line;
                    while (std::getline(file, line)) {
                        // Assuming each line contains a single float
                        float data;
                        std::istringstream iss(line);
                        if (iss >> data) {
                            fileData.push_back(data);
                        }
                        else {
                            std::cerr << "Error parsing data in file: " << fileName << std::endl;
                        }
                    }
                    file.close();
                }
                else {
                    std::cerr << "Error opening file: " << fileName << std::endl;
                }
                DataPoint datapoint(classNumber, fileData);
                //Check if class already exists in the vector
                dataset[classNumber].push_back(datapoint);
            }
            else {
                std::cerr << "Error parsing class number in file name: " << fileName << std::endl;
            }
        }
    }

    std::default_random_engine rng(seed);

    //Split each class using the given split ratio
    for (auto& [key, classVector] : dataset)
    {
        // shuffling the data
        std::shuffle(classVector.begin(), classVector.end(), rng);
        // split the data
        int trainingCount = int(classVector.size() * trainingRatio);
        for (int i = 0; i < trainingCount; ++i)
        {
            trainingData.push_back(classVector[i]);
        }
        for (int i = trainingCount; i < classVector.size(); ++i)
        {
            evaluationData.push_back(classVector[i]);
        }
    }

}

DatasetLoader::DatasetLoader(std::string folderPath, float trainingRatio)
    :
    DatasetLoader(folderPath, std::random_device()(), trainingRatio)
{
}

int DatasetLoader::GetClassCount() const
{
    return dataset.size();
}

const std::vector<DataPoint>& DatasetLoader::GetTrainingData() const
{
    return trainingData;
}

const std::vector<DataPoint>& DatasetLoader::GetEvaluationData() const
{
    return evaluationData;
}

DataPoint::DataPoint(int classIndex, std::vector<float> data)
    :
    classIndex(classIndex),
    data(data)
{
}
