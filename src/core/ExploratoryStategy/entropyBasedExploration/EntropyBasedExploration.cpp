#include  "EntropyBasedExploration.h"

EntropyBasedExploration::EntropyBasedExploration(float weight = 0.01f, float minWeight = 0.001f, float decay = 0.995f)
    : entropyWeight(weight), minEntropyWeight(minWeight), decay(decay) {
}

float EntropyBasedExploration::getExplorationRate(int step) const override {
    return minEntropyWeight + (entropyWeight - minEntropyWeight) * std::exp(-0.001f * step);
}

void EntropyBasedExploration::reset() override {
    entropyWeight = 0.01f;
}