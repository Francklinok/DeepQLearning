#include "BoltzmannExploration.h"

BoltzmannExploration::BoltzmannExploration(float startTemp = 1.0f, float minTemp = 0.1f, float decay = 0.995f)
	: temperature(startTemp), minTemp(minTemp), decay(decay) {}

float BoltzmannExploration::getExplorationRate(int step) const override {
    // Le taux retourné est la température actuelle
    float temp = minTemp + (temperature - minTemp) * std::exp(-0.0005f * step);
    return std::max(temp, minTemp);
}

void BoltzmannExploration::reset() override {
    temperature = 1.0f;
}
