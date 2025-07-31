#pragma once

#include "ExplorationStrategy.hpp"
#include <cmath>
#include <algorithm>
#include <memory>

/**
 * @brief Stratégie Epsilon-Greedy avec décroissance exponentielle
 * @tparam T Type de données pour les calculs
 * @tparam HardwareTarget Cible matérielle pour l'exécution
 */
template<typename T = float, typename HardwareTarget = CPU>
class EpsilonGreedy : public ExplorationStrategy<T, HardwareTarget> {
public:
    /**
     * @brief Constructeur avec paramètres configurables
     * @param start Valeur epsilon initiale
     * @param end Valeur epsilon minimale
     * @param decay Facteur de décroissance
     * @param minSteps Nombre minimal de pas avant décroissance
     */
    EpsilonGreedy(
        T start = static_cast<T>(1.0),
        T end = static_cast<T>(0.01),
        T decay = static_cast<T>(0.995),
        int64_t minSteps = 1000
    )
        : epsilonStart(start),
        epsilonEnd(end),
        decay(decay),
        minSteps(minSteps)
    {
        if (epsilonStart < epsilonEnd) {
            std::swap(epsilonStart, epsilonEnd);
        }
        if (decay <= 0 || decay >= 1) {
            this->decay = static_cast<T>(0.995);
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
            T cachedValue = context->getCachedValue(step, "epsilon");
            if (!std::isnan(cachedValue)) {
                return cachedValue;
            }
        }

        T result;
        if (step < minSteps) {
            result = epsilonStart;
        }
        else {
            result = epsilonEnd + (epsilonStart - epsilonEnd) *
                std::exp(-static_cast<T>(1.0) * static_cast<T>(step - minSteps) / (decay * static_cast<T>(minSteps)));
        }

        if (context) {
            context->setCachedValue(step, "epsilon", result);
        }
        return result;
    }

    /**
     * @brief Réinitialise la stratégie à son état initial
     */
    void reset() override {
        // Pas d'état interne dynamique, donc rien à réinitialiser
    }

    /**
     * @brief Ajuste les paramètres de la stratégie en fonction des métriques
     * @param metrics Métriques d'exploration
     */
    void adaptToMetrics(const typename ExplorationStrategy<T, HardwareTarget>::metrics_type& metrics) override {
        T explorationEfficiency = metrics.getExplorationEfficiency();
        T learningProgress = metrics.getLearningProgress();

        if (learningProgress > static_cast<T>(0.8)) {
            decay = std::max(decay * static_cast<T>(0.95), static_cast<T>(0.9));
        }
        else if (explorationEfficiency < static_cast<T>(0.3)) {
            decay = std::min(decay * static_cast<T>(1.05), static_cast<T>(0.999));
        }
    }

    /**
     * @brief Clone la stratégie
     * @return Pointeur unique vers une nouvelle instance
     */
    std::unique_ptr<ExplorationStrategy<T, HardwareTarget>> clone() const override {
        return std::make_unique<EpsilonGreedy<T, HardwareTarget>>(epsilonStart, epsilonEnd, decay, minSteps);
    }

#ifdef USE_GPU
    /**
     * @brief Version du calcul optimisée pour GPU
     */
    __device__ T getExplorationRateDevice(int64_t step) const override {
        if (step < minSteps) {
            return epsilonStart;
        }
        return epsilonEnd + (epsilonStart - epsilonEnd) *
            expf(-1.0f * static_cast<float>(step - minSteps) / (decay * static_cast<float>(minSteps)));
    }
#endif

private:
    T epsilonStart;   // Valeur epsilon initiale
    T epsilonEnd;     // Valeur epsilon minimale
    T decay;          // Facteur de décroissance
    int64_t minSteps; // Nombre minimal de pas avant décroissance
};

// Instanciations explicites pour les types courants
template class EpsilonGreedy<float, CPU>;
template class EpsilonGreedy<double, CPU>;

#ifdef USE_GPU
template class EpsilonGreedy<float, GPU>;
template class EpsilonGreedy<double, GPU>;
#endif
