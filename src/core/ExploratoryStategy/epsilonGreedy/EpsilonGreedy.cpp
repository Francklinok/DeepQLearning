#include  "EpsilonGreedy.h"

EpsilonGreedy::EpsilonGreedy(float start = 1.0f, float end = 0.01f, float decay = 0.995f, int  minSteps = 1000)
	:epsilonStart(start), epsilonEnd(end), decay(decay),  minSteps(minSteps) { }

float EpsilonGreedy::getExploratoryRate(int step)const override {
	if (step < minStep) return  epsilonStart;
	return epsilonEnd + (epsilonStart - epsilonEnd) * std::exp(-1.0f * step/(decay * minSteps))
}

void EpsilonGreedy::reset()  override {
	//
}