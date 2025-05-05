#pragma once
#include "ExplorationStrategy.h"

class NoisyExploration : public ExplorationStrategy {
private:
	float  noiseScale;
	float  decayRate;

public:
	NoisyExploration(float scale = 0.5f, float decay = 0.99f);
	float  getExploratorionRate(int step) const override;
	void reset() override;

};