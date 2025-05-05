#pragma  once
#include "../includes/core/ExploratoryStrategy/ExplorationStrategy.h"

class EpsilonGreedy :public ExplorationStrategy {
private:
	float  epsilonStart;
	float  epsilonEnd;
	float  decay;
	int  minSteps;

public:
	EpsilonGreedy(float start = 1.0f, float end = 0.01f, float decay = 0.995f, int  minSteps = 1000);

	float getExploratoryRate(int step)const override;
	void reset() override;
};