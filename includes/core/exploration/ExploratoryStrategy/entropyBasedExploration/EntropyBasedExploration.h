#pragma once

#include "ExplorationStrategy.hpp"
#include "DefaultEntropyCalculator.h"
#include <cmath>
#include <algorithm>
#include <memory>

/**
 * @brief Stratégie basée sur l'entropie avec ajustement adaptatif
 * @tparam T Type de données pour les calculs
 * @tparam HardwareTarget Cible matérielle pour l'exécution
 * @tparam EntropyCalculator Type de calculateur d'entropie
 */
template<
    typename T = float,
    typename HardwareTarget = CPU,
    template<typename> class EntropyCalculator = DefaultEntropyCalculator
>
class EntropyBasedExploration : public ExplorationStrategy<T, HardwareTarget> {
public:
    /**
     * @brief Constructeur avec paramètres configurables
     * @param weight Poids initial de l'entropie
     * @param minWeight Poids minimal de l'entropie
     * @param decay Facteur de décroissance
     * @param adaptiveFactor Facteur d'adaptation dynamique
     */
    EntropyBasedExploration(
        T weight = static_cast<T>(0.01),
        T minWeight = static_cast<T>(0.001),
        T decay = static_cast<T>(0.995),
        T adaptiveFactor = static_cast<T>(1.2)
    )
        : entropyWeight(weight)
        , minEntropyWeight(minWeight)
        , decay(decay)
        , initialWeight(weight)
        , adaptiveFactor(adaptiveFactor)
        , entropyCalculator()
    {
        if (entropyWeight < minEntropyWeight) {
            std::swap(entropyWeight, minEntropyWeight);
        }
        if (decay <= 0 || decay >= 1) {
            this->decay = static_cast<T>(0.995);
        }
        if (adaptiveFactor <= 1) {
            this->adaptiveFactor = static_cast<T>(1.2);
        }
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
            T cachedValue = context->getCachedValue(step, "entropy");
            if (!std::isnan(cachedValue)) {
                return cachedValue;
            }
        }

        T baseRate = minEntropyWeight + (entropyWeight - minEntropyWeight) *
            std::exp(-static_cast<T>(0.001) * static_cast<T>(step));

        if (context) {
            T entropyFactor = entropyCalculator.calculate(context);
            baseRate *= std::pow(entropyFactor, adaptiveFactor);
        }

        T result = std::clamp(baseRate, static_cast<T>(0), static_cast<T>(1));
        if (context) {
            context->setCachedValue(step, "entropy", result);
        }
        return result;
    }

    /**
     * @brief Réinitialise la stratégie à son état initial
     */
    void reset() override {
        entropyWeight = initialWeight;
    }

    /**
     * @brief Ajuste les paramètres de la stratégie en fonction des métriques
     * @param metrics Métriques d'exploration
     */
    void adaptToMetrics(const typename ExplorationStrategy<T, HardwareTarget>::metrics_type& metrics) override {
        T entropyLevel = metrics.getEntropyLevel();
        T learningProgress = metrics.getLearningProgress();

        if (entropyLevel < static_cast<T>(0.2) && learningProgress < static_cast<T>(0.5)) {
            entropyWeight = std::min(entropyWeight * adaptiveFactor, initialWeight * static_cast<T>(3.0));
        }
        else if (entropyLevel > static_cast<T>(0.8) || learningProgress > static_cast<T>(0.8)) {
            entropyWeight = std::max(entropyWeight / adaptiveFactor, minEntropyWeight);
        }
    }

    /**
     * @brief Clone la stratégie
     * @return Pointeur unique vers une nouvelle instance
     */
    std::unique_ptr<ExplorationStrategy<T, HardwareTarget>> clone() const override {
        return std::make_unique<EntropyBasedExploration<T, HardwareTarget, EntropyCalculator>>(
            entropyWeight, minEntropyWeight, decay, adaptiveFactor);
    }

#ifdef USE_GPU
    __device__ T getExplorationRateDevice(int64_t step) const override {
        T baseRate = minEntropyWeight + (entropyWeight - minEntropyWeight) *
            expf(-0.001f * static_cast<float>(step));
        return max(0.0f, min(1.0f, baseRate));
    }
#endif

private:
    mutable T entropyWeight;
    T minEntropyWeight;
    T decay;
    T initialWeight;
    T adaptiveFactor;
    EntropyCalculator<T> entropyCalculator;
};

// Instanciations explicites pour les types courants
template class EntropyBasedExploration<float, CPU, DefaultEntropyCalculator>;
template class EntropyBasedExploration<double, CPU, DefaultEntropyCalculator>;

#ifdef USE_GPU
template class EntropyBasedExploration<float, GPU, DefaultEntropyCalculator>;
template class EntropyBasedExploration<double, GPU, DefaultEntropyCalculator>;
#endif
