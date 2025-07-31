#pragma once

#include "ExplorationStrategy.hpp"
#include <random>
#include <chrono>
#include <algorithm>
#include <cmath>
#include <memory>

/**
 * @brief Stratégie d'exploration basée sur le bruit
 * @tparam T Type de données pour les calculs
 * @tparam HardwareTarget Cible matérielle pour l'exécution
 * @tparam NoiseDistribution Type de distribution du bruit (Normal, Uniform, etc.)
 */
template<
    typename T = float,
    typename HardwareTarget = CPU,
    typename NoiseDistribution = std::normal_distribution<double>
>
class NoisyExploration : public ExplorationStrategy<T, HardwareTarget> {
public:
    /**
     * @brief Constructeur avec paramètres configurables
     * @param scale Échelle initiale du bruit
     * @param decay Taux de décroissance du bruit
     * @param adaptiveNoise Activer l'ajustement adaptatif du bruit
     */
    NoisyExploration(
        T scale = static_cast<T>(0.5),
        T decay = static_cast<T>(0.99),
        bool adaptiveNoise = true
    )
        : noiseScale(scale),
        decayRate(decay),
        initialScale(scale),
        adaptiveNoiseEnabled(adaptiveNoise),
        lastResetTimestamp(std::chrono::steady_clock::now())
    {
        if (noiseScale < 0) noiseScale = std::abs(noiseScale);
        if (decayRate <= 0 || decayRate > 1) decayRate = static_cast<T>(0.99);
    }

    /**
     * @brief Obtient le taux d'exploration à l'étape spécifiée
     * @param step Étape actuelle
     * @param context Contexte d'exécution optionnel
     * @return Taux d'exploration
     */
    T getExplorationRate(
        int64_t step,
        const typename ExplorationStrategy<T, HardwareTarget>::context_type* context = nullptr
    ) const override {
        if (context) {
            T cachedValue = context->getCachedValue(step, "noisy");
            if (!std::isnan(cachedValue)) return cachedValue;
        }

        T baseRate = noiseScale * std::pow(decayRate, static_cast<T>(step) / static_cast<T>(1000));
        T noise = 0;

        if (context) {
            noise = context->generateNormalRandom() * baseRate * static_cast<T>(0.1);
        }
        else {
            std::normal_distribution<double> dist(0, 0.1);
            std::mt19937 engine(static_cast<unsigned int>(std::hash<int64_t>{}(step)));
            noise = static_cast<T>(dist(engine)) * baseRate;
        }

        T result = std::clamp(baseRate + noise, static_cast<T>(0), static_cast<T>(1));

        if (context) {
            context->setCachedValue(step, "noisy", result);
        }

        return result;
    }

    /**
     * @brief Réinitialise la stratégie à son état initial
     */
    void reset() override {
        noiseScale = initialScale;
        lastResetTimestamp = std::chrono::steady_clock::now();
    }

    /**
     * @brief Ajuste les paramètres de la stratégie en fonction des métriques
     * @param metrics Métriques d'exploration
     */
    void adaptToMetrics(const typename ExplorationStrategy<T, HardwareTarget>::metrics_type& metrics) override {
        if (!adaptiveNoiseEnabled) return;

        T explorationEfficiency = metrics.getExplorationEfficiency();
        T entropyLevel = metrics.getEntropyLevel();

        if (entropyLevel < static_cast<T>(0.3)) {
            noiseScale = std::min(noiseScale * static_cast<T>(1.05), initialScale * static_cast<T>(2.0));
        }
        else if (explorationEfficiency < static_cast<T>(0.4)) {
            noiseScale = std::max(noiseScale * static_cast<T>(0.98), initialScale * static_cast<T>(0.1));
        }
    }

    /**
     * @brief Clone la stratégie
     * @return Pointeur unique vers une nouvelle instance
     */
    std::unique_ptr<ExplorationStrategy<T, HardwareTarget>> clone() const override {
        return std::make_unique<NoisyExploration<T, HardwareTarget, NoiseDistribution>>(
            noiseScale, decayRate, adaptiveNoiseEnabled);
    }

#ifdef USE_GPU
    /**
     * @brief Version du calcul optimisée pour GPU
     */
    __device__ T getExplorationRateDevice(int64_t step) const override {
        T baseRate = noiseScale * powf(decayRate, static_cast<float>(step) / 1000.0f);

        unsigned int seed = step * 1099087573u;
        seed = (seed ^ 61) ^ (seed >> 16);
        seed *= 9;
        seed = seed ^ (seed >> 4);
        seed *= 0x27d4eb2d;
        seed = seed ^ (seed >> 15);

        float randVal = static_cast<float>(seed) / static_cast<float>(UINT_MAX);
        randVal = (randVal * 2.0f) - 1.0f;  // -1 à 1

        T noise = static_cast<T>(randVal) * baseRate * static_cast<T>(0.1);
        return max(0.0f, min(1.0f, baseRate + noise));
    }
#endif

private:
    mutable T noiseScale;
    T decayRate;
    T initialScale;
    bool adaptiveNoiseEnabled;
    mutable std::chrono::steady_clock::time_point lastResetTimestamp;
};

// Instanciations explicites
template class NoisyExploration<float, CPU, std::normal_distribution<double>>;
template class NoisyExploration<double, CPU, std::normal_distribution<double>>;
template class NoisyExploration<float, CPU, std::uniform_real_distribution<double>>;
template class NoisyExploration<double, CPU, std::uniform_real_distribution<double>>;

#ifdef USE_GPU
template class NoisyExploration<float, GPU, std::normal_distribution<double>>;
template class NoisyExploration<double, GPU, std::normal_distribution<double>>;
#endif
