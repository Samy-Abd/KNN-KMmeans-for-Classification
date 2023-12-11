#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <random>
#include <cstdlib>

namespace fs = std::filesystem;

struct FileInfo {
    std::vector<float> fileData;
    int classNumber;
};

struct DataPoint {
    int centroidNumber;
    int classNumber;
};


void processFilesInFolder(const std::string& folderPath, std::vector<FileInfo>& fileInfos) {
    for (const auto& entry : fs::directory_iterator(folderPath)) {
        if (entry.is_regular_file()) {
            // Get the file name from the path
            std::string fileName = entry.path().filename().string();

            // Parse file name to get class number
            int classNumber;
            if (sscanf_s(fileName.c_str(), "s%d", &classNumber) == 1 || sscanf_s(fileName.c_str(), "S%d", &classNumber) == 1) {
                FileInfo fileInfo;

                fileInfo.classNumber = classNumber;

                // Read data from the file
                std::ifstream file(entry.path());
                if (file.is_open()) {
                    std::string line;
                    while (std::getline(file, line)) {
                        // Assuming each line contains a single float
                        float data;
                        std::istringstream iss(line);
                        if (iss >> data) {
                            fileInfo.fileData.push_back(data);
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

                // Add FileInfo to the vector
                fileInfos.push_back(fileInfo);
            }
            else {
                std::cerr << "Error parsing class number in file name: " << fileName << std::endl;
            }
        }
    }
}

// Function to calculate Euclidean distance between two data vectors
float calculateDistance(const std::vector<float>& point1, const std::vector<float>& point2) {
    float distance = 0.0;
    for (size_t i = 0; i < point1.size(); ++i) {
        distance += pow(point1[i] - point2[i], 2);
    }
    return sqrt(distance);
}


// Function to find k-nearest neighbors for a given data vector
std::vector<int> findNeighbors(int k, const std::vector<float>& queryData, const std::vector<FileInfo>& trainingData) {
    std::vector<std::pair<float, int>> distances;

    for (const auto& fileInfo : trainingData) {
        float distance = calculateDistance(queryData, fileInfo.fileData);
        distances.push_back({ distance, fileInfo.classNumber });
    }

    // Sort distances in ascending order
    std::sort(distances.begin(), distances.end());

    // Return the classes of the first k neighbors
    std::vector<int> neighborsClasses;
    for (int i = 0; i < k; ++i) {
        neighborsClasses.push_back(distances[i].second);
    }

    return neighborsClasses;
}


// Function to perform majority voting among the neighbors
int majorityVote(const std::vector<int>& neighborsClasses) {
    // Count occurrences of each class
    std::vector<int> classCount(*std::max_element(neighborsClasses.begin(), neighborsClasses.end()) + 1, 0);

    for (int neighborClass : neighborsClasses) {
        classCount[neighborClass]++;
    }

    // Find the class with the maximum count
    int maxCount = -1;
    int predictedClass = -1;

    for (size_t i = 0; i < classCount.size(); ++i) {
        if (classCount[i] > maxCount) {
            maxCount = classCount[i];
            predictedClass = static_cast<int>(i);
        }
    }

    return predictedClass;
}


// KNN function
int knn(int k, const std::vector<float>& queryData, const std::vector<FileInfo>& trainingData) {
    std::vector<int> neighborsClasses = findNeighbors(k, queryData, trainingData);
    return majorityVote(neighborsClasses);
}


std::vector<FileInfo> kMeansClustering(int k, std::vector<FileInfo>& data) {
    if (data.empty()) {
        std::cerr << "Error: No data provided." << std::endl;
        return {};
    }

    // generating a random device to randomize the data vector
    std::random_device rd;
    std::default_random_engine rng(rd());

    // shuffling the data
    std::vector<FileInfo> shfl = data;
    std::shuffle(shfl.begin(), shfl.end(), rng);

    // Initialize centroids with the first k data points
    std::vector<FileInfo> centroids(shfl.begin(), shfl.begin() + k);

    // Number of iterations (you may adjust this based on your requirements)
    const int maxIterations = 100;

    for (int iteration = 0; iteration < maxIterations; ++iteration) {
        // Assign each data point to the nearest centroid
        for (FileInfo& point : data) {
            float minDistance = std::numeric_limits<float>::max();
            int nearestCentroidIndex = -1;

            for (size_t i = 0; i < centroids.size(); ++i) {
                float distance = calculateDistance(point.fileData, centroids[i].fileData);
                if (distance < minDistance) {
                    minDistance = distance;
                    nearestCentroidIndex = i;
                }
            }

            point.classNumber = nearestCentroidIndex;
        }

        // Update centroids based on the mean of assigned data points
        for (size_t i = 0; i < centroids.size(); ++i) {
            std::vector<float> sum(centroids[i].fileData.size(), 0.0);
            int count = 0;

            for (const FileInfo& point : data) {
                if (point.classNumber == static_cast<int>(i)) {
                    for (size_t j = 0; j < sum.size(); ++j) {
                        sum[j] += point.fileData[j];
                    }
                    ++count;
                }
            }

            if (count > 0) {
                for (size_t j = 0; j < sum.size(); ++j) {
                    centroids[i].fileData[j] = sum[j] / count;
                }
            }
        }
    }

    return centroids;
}


int kMeansPred(const std::vector<float>& dataPoint, const std::vector<FileInfo>& centroids) {
    if (centroids.empty()) {
        std::cerr << "Error: No centroids provided." << std::endl;
        return -1;  // Return an error code
    }

    float minDistance = std::numeric_limits<float>::max();
    int predictedClass = -1;

    for (size_t i = 0; i < centroids.size(); ++i) {
        float distance = calculateDistance(dataPoint, centroids[i].fileData);
        if (distance < minDistance) {
            minDistance = distance;
            predictedClass = int(i);
        }
    }

    return predictedClass;
}



int main() {

    std::vector<float> E34_1 = { 0.222689, 0.283041, 0.335261, 0.380889, 0.421783, 0.459126, 0.493693, 0.526017, 0.556481, 0.585372,
    0.612977, 0.725527, 0.747724, 0.808657, 0.935889, 1.312580 }; // class 9

    std::vector<float> E34_2 = { 0.180883, 0.248000, 0.329304, 0.372480, 0.411110, 0.446412, 0.492422, 0.522361, 0.550699, 0.577650,
      0.673020, 0.726571, 0.828345, 0.883082, 1.022771,1.312201 }; // class 1

    std::vector<float> E34_3 = { 0.284875, 0.419235, 0.506243, 0.600408, 0.666742, 0.743160, 0.798370, 0.868427, 0.936294, 0.999888,
        1.111963, 1.183781, 1.302885, 1.493834, 1.817366, 2.746209 }; // class 3


    std::vector<float> GFD_1 = {
        0.262132, 0.031934, 0.375073, 0.226145, 0.281254, 0.122168, 0.094361, 0.090860,
        0.080594, 0.032331, 0.308207, 0.015002, 0.091666, 0.218990, 0.094148, 0.092455,
        0.112666, 0.132173, 0.048330, 0.047028, 0.154325, 0.068969, 0.024158, 0.024936,
        0.029745, 0.021462, 0.061306, 0.047188, 0.034495, 0.056849, 0.082687, 0.024045,
        0.020806, 0.067385, 0.009291, 0.033248, 0.039949, 0.028890, 0.020963, 0.033967,
        0.019985, 0.028931, 0.020096, 0.038447, 0.016817, 0.033104, 0.036361, 0.016807,
        0.019434, 0.015903, 0.019795, 0.004791, 0.027976, 0.004910, 0.019603, 0.025939,
        0.010427, 0.015360, 0.021733, 0.008446, 0.017548, 0.005846, 0.011612, 0.016917,
        0.021949, 0.007206, 0.020297, 0.003163, 0.019889, 0.007178, 0.004788, 0.007710,
        0.022266, 0.002980, 0.006366, 0.015695, 0.005012, 0.004587, 0.006811, 0.011256,
        0.012484, 0.008555, 0.005734, 0.014866, 0.013362, 0.013133, 0.014118, 0.007682,
        0.015704, 0.002310, 0.008091, 0.013417, 0.012974, 0.015966, 0.005697, 0.012657,
        0.003250, 0.001630, 0.004381, 0.005157
    }; // class 1

    std::vector<float> GFD_2 = {
        0.432006, 0.051541, 0.136565, 0.064174, 0.078550, 0.101208, 0.026701, 0.042862,
        0.032370, 0.010623, 0.446037, 0.161989, 0.037809, 0.075633, 0.101408, 0.124750,
        0.054058, 0.050940, 0.024090, 0.030798, 0.120536, 0.131315, 0.030720, 0.102133,
        0.052466, 0.071465, 0.026282, 0.024214, 0.045011, 0.019076, 0.068709, 0.021099,
        0.017886, 0.063681, 0.023594, 0.024196, 0.018860, 0.015291, 0.050670, 0.020461,
        0.031477, 0.014681, 0.018567, 0.032385, 0.037518, 0.014243, 0.007588, 0.013035,
        0.015320, 0.016680, 0.027439, 0.001919, 0.004387, 0.029978, 0.021525, 0.007517,
        0.006592, 0.014930, 0.015899, 0.035220, 0.015017, 0.017558, 0.000724, 0.020100,
        0.014137, 0.011533, 0.010714, 0.016335, 0.015482, 0.020895, 0.019312, 0.012504,
        0.006560, 0.016237, 0.010023, 0.007487, 0.012573, 0.005428, 0.004442, 0.011809,
        0.011924, 0.011741, 0.011625, 0.008427, 0.001636, 0.007131, 0.005912, 0.006676,
        0.007377, 0.009168, 0.001377, 0.010064, 0.006189, 0.011956, 0.002592, 0.008376,
        0.004634, 0.011791, 0.004667, 0.005507
    }; // class 5

    // Example usage
    std::string folderPath = "C://Users//caggi//OneDrive//Bureau//test";
    std::vector<FileInfo> fileInfos;

    processFilesInFolder(folderPath, fileInfos);

    for (const auto& dataPoint : fileInfos) {
        std::cout << "Centroid Number: " << dataPoint.classNumber << ", ";
        std::cout << " ***** " << std::endl;
    }



    /*int cls = knn(10, GFD_2, fileInfos);

        std::cout << cls << std::endl;

    std::vector<FileInfo> finalCentroids = kMeansClustering(9, fileInfos);


    int predictedClass = kMeansPred(GFD_2, finalCentroids);

        std::cout << "Predicted cluster: " << predictedClass << std::endl;

    
    std::vector<DataPoint> checks;

    for (const auto& fileInfo : fileInfos) {
    // Access the class number and the data vector
    int classNumber = fileInfo.classNumber;
    std::vector<float> dataVector = fileInfo.fileData;


    std::cout << " Class number" << " ";
    std::cout << classNumber << std::endl;
    for (const auto& element : dataVector) {
        std::cout << element << ", ";
    }
    std::cout << std::endl;

    int preds = kMeansPred(dataVector, finalCentroids);

    DataPoint p;
    p.classNumber = classNumber;
    p.centroidNumber = preds;

    checks.emplace_back(p);
    }

    for (const auto& dataPoint : checks) {
        std::cout << "Centroid Number: " << dataPoint.centroidNumber << ", ";
        std::cout << "Class Number: " << dataPoint.classNumber << std::endl;
        std::cout << " ***** " << std::endl;
    }*/



    return 0;
}
