#include "NoisyExploration.h"

NoisyExploration::NoisyExploration(float scale = 0.5f, float decay = 0.99f)
	:noiseScale(scale), decayRate(decay){ }


float  NoisyExploration::getExploratorionRate(int step) const override {
return noiseScale + std::pow(decayRate, step/1000)
}

void NoisyExploration::reset() override {
	return noiseScale = 0.5f
}