#pragma once
# include "ExplorationStrategy.h"

class EntropyBasedExploration : public ExplorationStrategy {
private:
    float entropyWeight;
    float minEntropyWeight;
    float decay;
public:
    EntropyBasedExploration(float weight = 0.01f, float minWeight = 0.001f, float decay = 0.995f);
    float getExplorationRate(int step) const override;
    void reset() override;
};