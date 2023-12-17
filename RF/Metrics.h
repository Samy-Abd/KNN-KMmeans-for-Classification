#pragma once
#include <vector>
#include <iostream>
using ConfusionMatrix = std::vector<std::vector<int>>;

struct PrecisionRecallF1
{
	float precision;
	float recall;
	float f1Score;
};
struct Metrics
{
	float accuracy;
	float time;
	ConfusionMatrix confusionMatrix;
	std::vector<PrecisionRecallF1> classesPrecisionRecallF1;
};


void PrintConfusionMatrix(const ConfusionMatrix& confusionMatrix);

void PrintMetrics(const Metrics& metrics);
