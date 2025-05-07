#include "NoisyExploration.h"
#include <cmath>
#include <functional>
#include <limits>
#include <algorithm>

template<
    typename T,
    typename HardwareTarget,
    typename NoiseDistribution
>
NoisyExploration<T, HardwareTarget, NoiseDistribution>::NoisyExploration(
    T scale,
    T decay,
    bool adaptiveNoise
)
    : noiseScale(scale)
    , decayRate(decay)
    , initialScale(scale)
    , adaptiveNoiseEnabled(adaptiveNoise)
    , lastResetTimestamp(std::chrono::steady_clock::now())
{
    // Validation des paramètres
    if (noiseScale < 0) {
        noiseScale = std::abs(noiseScale);
    }

    if (decayRate <= 0 || decayRate > 1) {
        decayRate = static_cast<T>(0.99);
    }
}

template<
    typename T,
    typename HardwareTarget,
    typename NoiseDistribution
>
T NoisyExploration<T, HardwareTarget, NoiseDistribution>::getExplorationRate(
    int64_t step,
    const typename ExplorationStrategy<T, HardwareTarget>::context_type* context
) const {
    // Utiliser le cache si disponible
    if (context) {
        T cachedValue = context->getCachedValue(step, "noisy");
        if (!std::isnan(cachedValue)) {
            return cachedValue;
        }
    }

    // Calcul du taux de base avec décroissance
    T baseRate = noiseScale * std::pow(decayRate, static_cast<T>(step) / static_cast<T>(1000));

    // Ajout de bruit stochastique pour éviter les minima locaux
    T noise = 0;
    if (context) {
        noise = context->generateNormalRandom() * baseRate * static_cast<T>(0.1);
    }
    else {
        std::normal_distribution<double> dist(0, 0.1);
        std::mt19937 engine(static_cast<unsigned int>(std::hash<int64_t>{}(step)));
        noise = static_cast<T>(dist(engine)) * baseRate;
    }

    // Limiter le taux final entre 0 et 1
    T result = std::clamp(baseRate + noise, static_cast<T>(0), static_cast<T>(1));

    // Mettre en cache le résultat
    if (context) {
        context->setCachedValue(step, "noisy", result);
    }

    return result;
}

template<
    typename T,
    typename HardwareTarget,
    typename NoiseDistribution
>
void NoisyExploration<T, HardwareTarget, NoiseDistribution>::reset() {
    noiseScale = initialScale;
    lastResetTimestamp = std::chrono::steady_clock::now();
}

template<
    typename T,
    typename HardwareTarget,
    typename NoiseDistribution
>
void NoisyExploration<T, HardwareTarget, NoiseDistribution>::adaptToMetrics(
    const typename ExplorationStrategy<T, HardwareTarget>::metrics_type& metrics
) {
    if (!adaptiveNoiseEnabled) return;

    T explorationEfficiency = metrics.getExplorationEfficiency();
    T entropyLevel = metrics.getEntropyLevel();

    // Ajuster dynamiquement l'échelle du bruit en fonction de l'entropie
    if (entropyLevel < static_cast<T>(0.3)) {
        // Augmenter le bruit si l'entropie est faible
        noiseScale = std::min(noiseScale * static_cast<T>(1.05), initialScale * static_cast<T>(2.0));
    }
    else if (explorationEfficiency < static_cast<T>(0.4)) {
        // Réduire le bruit si l'exploration est inefficace
        noiseScale = std::max(noiseScale * static_cast<T>(0.98), initialScale * static_cast<T>(0.1));
    }
}

template<
    typename T,
    typename HardwareTarget,
    typename NoiseDistribution
>
std::unique_ptr<ExplorationStrategy<T, HardwareTarget>>
NoisyExploration<T, HardwareTarget, NoiseDistribution>::clone() const {
    return std::make_unique<NoisyExploration<T, HardwareTarget, NoiseDistribution>>(
        noiseScale, decayRate, adaptiveNoiseEnabled);
}

#ifdef USE_GPU
template<
    typename T,
    typename HardwareTarget,
    typename NoiseDistribution
>
__device__ T NoisyExploration<T, HardwareTarget, NoiseDistribution>::getExplorationRateDevice(int64_t step) const {
    // Version simplifiée pour GPU sans générateur de nombres aléatoires
    T baseRate = noiseScale * powf(decayRate, static_cast<float>(step) / 1000.0f);

    // Utiliser un générateur pseudo-aléatoire simple basé sur l'étape
    unsigned int seed = step * 1099087573u;
    seed = (seed ^ 61) ^ (seed >> 16);
    seed *= 9;
    seed = seed ^ (seed >> 4);
    seed *= 0x27d4eb2d;
    seed = seed ^ (seed >> 15);

    float randVal = static_cast<float>(seed) / static_cast<float>(UINT_MAX);
    randVal = (randVal * 2.0f) - 1.0f;  // -1 à 1

    T noise = static_cast<T>(randVal) * baseRate * static_cast<T>(0.1);

    // Limiter entre 0 et 1
    return max(0.0f, min(1.0f, baseRate + noise));
}
#endif

// Explicitement instancier les templates pour les types courants
template class NoisyExploration<float, CPU, std::normal_distribution<double>>;
template class NoisyExploration<double, CPU, std::normal_distribution<double>>;
template class NoisyExploration<float, CPU, std::uniform_real_distribution<double>>;
template class NoisyExploration<double, CPU, std::uniform_real_distribution<double>>;

#ifdef USE_GPU
template class NoisyExploration<float, GPU, std::normal_distribution<double>>;
template class NoisyExploration<double, GPU, std::normal_distribution<double>>;
#endif