#include "DatasetLoader.h"
#include <filesystem>

int main()
{
	std::filesystem::path currentPath = std::filesystem::current_path();
	DatasetLoader datasetLoader{ currentPath.string() + "\\..\\images\\E34"};
	return 0;
}