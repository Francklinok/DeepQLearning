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
    );

    /**
     * @brief Obtient le taux d'exploration à l'étape spécifiée
     * @param step Étape actuelle
     * @param context Contexte d'exécution optionnel
     * @return Taux d'exploration
     */
    T getExplorationRate(int64_t step, const typename ExplorationStrategy<T, HardwareTarget>::context_type* context = nullptr) const override;

    /**
     * @brief Réinitialise la stratégie à son état initial
     */
    void reset() override;

    /**
     * @brief Ajuste les paramètres de la stratégie en fonction des métriques
     * @param metrics Métriques d'exploration
     */
    void adaptToMetrics(const typename ExplorationStrategy<T, HardwareTarget>::metrics_type& metrics) override;

    /**
     * @brief Clone la stratégie
     * @return Pointeur unique vers une nouvelle instance
     */
    std::unique_ptr<ExplorationStrategy<T, HardwareTarget>> clone() const override;

#ifdef USE_GPU
    /**
     * @brief Version du calcul optimisée pour GPU
     */
    __device__ T getExplorationRateDevice(int64_t step) const override;
#endif

private:
    T epsilonStart;  // Valeur epsilon initiale
    T epsilonEnd;    // Valeur epsilon minimale
    T decay;         // Facteur de décroissance
    int64_t minSteps; // Nombre minimal de pas avant décroissance
};