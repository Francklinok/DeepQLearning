#pragma once

#include "ExplorationStrategy.hpp"
#include "DefaultEntropyCalculator.h"
#include <cmath>
#include <algorithm>
#include <memory>

/**
 * @brief Strat�gie bas�e sur l'entropie avec ajustement adaptatif
 * @tparam T Type de donn�es pour les calculs
 * @tparam HardwareTarget Cible mat�rielle pour l'ex�cution
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
     * @brief Constructeur avec param�tres configurables
     * @param weight Poids initial de l'entropie
     * @param minWeight Poids minimal de l'entropie
     * @param decay Facteur de d�croissance
     * @param adaptiveFactor Facteur d'adaptation dynamique
     */
    EntropyBasedExploration(
        T weight = static_cast<T>(0.01),
        T minWeight = static_cast<T>(0.001),
        T decay = static_cast<T>(0.995),
        T adaptiveFactor = static_cast<T>(1.2)
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
    mutable T entropyWeight;    // Poids de l'entropie (mutable pour adaptation)
    T minEntropyWeight;        // Poids minimal de l'entropie
    T decay;                   // Facteur de d�croissance
    T initialWeight;           // Poids initial pour reset
    T adaptiveFactor;          // Facteur d'adaptation dynamique

    // Calculateur d'entropie
    EntropyCalculator<T> entropyCalculator;
};