#pragma once

#include <memory>
#include <cstdint>
#include <type_traits>
#include "ExplorationMetrics.h"
#include "ExecutionContext.h"
#include "HardwareTraits.h"

/**
 * @brief Classe de base abstraite pour toutes les stratégies d'exploration.
 *
 * @tparam T Type de données flottant (float, double, etc.)
 * @tparam HardwareTarget Cible matérielle (CPU ou GPU)
 */
template <typename T = float, typename HardwareTarget = CPU>
class ExplorationStrategy {
public:
    using value_type = T;
    using metrics_type = ExplorationMetrics<T>;
    using context_type = ExecutionContext<T>;

    static_assert(std::is_floating_point_v<T>, "Le type doit être à virgule flottante");
    static_assert(hardware_available<HardwareTarget>::value, "Hardware target non disponible");

    virtual ~ExplorationStrategy() = default;

    /**
     * @brief Calcule le taux d'exploration pour une étape donnée.
     * @param step Étape actuelle (itération).
     * @param context Contexte d'exécution optionnel.
     * @return Taux d'exploration (entre 0 et 1).
     */
    virtual T getExplorationRate(int64_t step, const context_type* context = nullptr) const = 0;

    /**
     * @brief Réinitialise la stratégie à son état initial.
     */
    virtual void reset() = 0;

    /**
     * @brief Ajuste dynamiquement les paramètres de la stratégie selon des métriques d'exploration.
     * @param metrics Ensemble de métriques.
     */
    virtual void adaptToMetrics(const metrics_type& metrics) {}

    /**
     * @brief Crée une copie polymorphique de la stratégie.
     * @return Pointeur unique vers une nouvelle instance.
     */
    virtual std::unique_ptr<ExplorationStrategy<T, HardwareTarget>> clone() const = 0;

protected:
#ifdef USE_GPU
    /**
     * @brief Version GPU du calcul du taux d'exploration.
     * @param step Étape actuelle.
     * @return Taux d'exploration.
     */
    __device__ virtual T getExplorationRateDevice(int64_t step) const = 0;
#endif
};
