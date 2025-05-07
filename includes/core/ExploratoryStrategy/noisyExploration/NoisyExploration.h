#pragma once

#include "ExplorationStrategy.hpp"
#include <random>
#include <chrono>
#include <algorithm>
#include <cmath>
#include <memory>

/**
 * @brief Strat�gie d'exploration bas�e sur le bruit
 * @tparam T Type de donn�es pour les calculs
 * @tparam HardwareTarget Cible mat�rielle pour l'ex�cution
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
     * @brief Constructeur avec param�tres configurables
     * @param scale �chelle initiale du bruit
     * @param decay Taux de d�croissance du bruit
     * @param adaptiveNoise Activer l'ajustement adaptatif du bruit
     */
    NoisyExploration(
        T scale = static_cast<T>(0.5),
        T decay = static_cast<T>(0.99),
        bool adaptiveNoise = true
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
    mutable T noiseScale;  // �chelle du bruit (mutable pour adaptation dynamique)
    T decayRate;          // Taux de d�croissance
    T initialScale;       // �chelle initiale pour reset
    bool adaptiveNoiseEnabled; // Activation de l'ajustement adaptatif

    // Horodatage pour fonctionnalit�s temporelles
    mutable std::chrono::steady_clock::time_point lastResetTimestamp;
};