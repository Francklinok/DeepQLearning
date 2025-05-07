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
    );

    /**
     * @brief Obtient le taux d'exploration � l'�tape sp�cifi�e
     * @param step �tape actuelle
     * @param context Contexte d'ex�cution optionnel
     * @return Taux d'exploration
     */
    T getExplorationRate(int64_t step, const typename ExplorationStrategy<T, HardwareTarget>::context_type* context = nullptr) const override;

    /**
     * @brief R�initialise la strat�gie � son �tat initial
     */
    void reset() override;

    /**
     * @brief Ajuste les param�tres de la strat�gie en fonction des m�triques
     * @param metrics M�triques d'exploration
     */
    void adaptToMetrics(const typename ExplorationStrategy<T, HardwareTarget>::metrics_type& metrics) override;

    /**
     * @brief Clone la strat�gie
     * @return Pointeur unique vers une nouvelle instance
     */
    std::unique_ptr<ExplorationStrategy<T, HardwareTarget>> clone() const override;

#ifdef USE_GPU
    /**
     * @brief Version du calcul optimis�e pour GPU
     */
    __device__ T getExplorationRateDevice(int64_t step) const override;
#endif

private:
    T epsilonStart;  // Valeur epsilon initiale
    T epsilonEnd;    // Valeur epsilon minimale
    T decay;         // Facteur de d�croissance
    int64_t minSteps; // Nombre minimal de pas avant d�croissance
};