#include "DatasetLoader.h"
#include <filesystem>
#include <iostream>
#include <fstream>
#include <sstream>

DatasetLoader::DatasetLoader(std::string folderPath, float split)
{
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
}

DataPoint::DataPoint(int classIndex, std::vector<float> data)
    :
    classIndex(classIndex),
    data(data)
{
}
