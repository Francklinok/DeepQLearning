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
    mutable T noiseScale;  // Échelle du bruit (mutable pour adaptation dynamique)
    T decayRate;          // Taux de décroissance
    T initialScale;       // Échelle initiale pour reset
    bool adaptiveNoiseEnabled; // Activation de l'ajustement adaptatif

    // Horodatage pour fonctionnalités temporelles
    mutable std::chrono::steady_clock::time_point lastResetTimestamp;
};