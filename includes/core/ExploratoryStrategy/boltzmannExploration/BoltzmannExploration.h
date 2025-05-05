#pragma once
#include  "ExplorationStrategy.h"


class BoltzmannExploration : public ExplorationStrategy {
private:
	float  temperature;
	float  minTemp;
	float  decay;

public:
	BoltzmannExploration(float startTemp = 1.0f, float minTemp = 0.1f, float decay = 0.995f);
	float getExplorationRate(int step) const override;
	void reset() override;
};