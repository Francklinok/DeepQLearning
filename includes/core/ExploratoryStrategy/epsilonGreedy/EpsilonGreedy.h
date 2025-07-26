#pragma once

#include "ExplorationStrategy.hpp"
#include <cmath>
#include <algorithm>
#include <memory>

/**
 * @brief Strat�gie Epsilon-Greedy avec d�croissance exponentielle
 * @tparam T Type de donn�es pour les calculs
 * @tparam HardwareTarget Cible mat�rielle pour l'ex�cution
 */
template<typename T = float, typename HardwareTarget = CPU>
class EpsilonGreedy : public ExplorationStrategy<T, HardwareTarget> {
public:
    /**
     * @brief Constructeur avec param�tres configurables
     * @param start Valeur epsilon initiale
     * @param end Valeur epsilon minimale
     * @param decay Facteur de d�croissance
     * @param minSteps Nombre minimal de pas avant d�croissance
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
     * @brief Obtient le taux d'exploration � l'�tape sp�cifi�e
     * @param step �tape actuelle
     * @param context Contexte d'ex�cution optionnel
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
     * @brief R�initialise la strat�gie � son �tat initial
     */
    void reset() override {
        // Pas d'�tat interne dynamique, donc rien � r�initialiser
    }

    /**
     * @brief Ajuste les param�tres de la strat�gie en fonction des m�triques
     * @param metrics M�triques d'exploration
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
     * @brief Clone la strat�gie
     * @return Pointeur unique vers une nouvelle instance
     */
    std::unique_ptr<ExplorationStrategy<T, HardwareTarget>> clone() const override {
        return std::make_unique<EpsilonGreedy<T, HardwareTarget>>(epsilonStart, epsilonEnd, decay, minSteps);
    }

#ifdef USE_GPU
    /**
     * @brief Version du calcul optimis�e pour GPU
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
    T decay;          // Facteur de d�croissance
    int64_t minSteps; // Nombre minimal de pas avant d�croissance
};

// Instanciations explicites pour les types courants
template class EpsilonGreedy<float, CPU>;
template class EpsilonGreedy<double, CPU>;

#ifdef USE_GPU
template class EpsilonGreedy<float, GPU>;
template class EpsilonGreedy<double, GPU>;
#endif
