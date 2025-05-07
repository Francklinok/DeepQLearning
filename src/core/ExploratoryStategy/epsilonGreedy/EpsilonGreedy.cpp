#include "EpsilonGreedy.h"
#include <cmath>
#include <algorithm>

template<typename T, typename HardwareTarget>
EpsilonGreedy<T, HardwareTarget>::EpsilonGreedy(
    T start,
    T end,
    T decay,
    int64_t minSteps
)
    : epsilonStart(start)
    , epsilonEnd(end)
    , decay(decay)
    , minSteps(minSteps)
{
    // Validation des paramètres
    if (epsilonStart < epsilonEnd) {
        std::swap(epsilonStart, epsilonEnd);
    }

    if (decay <= 0 || decay >= 1) {
        decay = static_cast<T>(0.995);
    }
}

template<typename T, typename HardwareTarget>
T EpsilonGreedy<T, HardwareTarget>::getExplorationRate(
    int64_t step,
    const typename ExplorationStrategy<T, HardwareTarget>::context_type* context
) const {
    // Utiliser le cache si disponible
    if (context) {
        T cachedValue = context->getCachedValue(step, "epsilon");
        if (!std::isnan(cachedValue)) {
            return cachedValue;
        }
    }

    // Calcul du taux avec transition douce
    T result;
    if (step < minSteps) {
        result = epsilonStart;
    }
    else {
        result = epsilonEnd + (epsilonStart - epsilonEnd) *
            std::exp(-static_cast<T>(1.0) * static_cast<T>(step - minSteps) / (decay * static_cast<T>(minSteps)));
    }

    // Mettre en cache le résultat
    if (context) {
        context->setCachedValue(step, "epsilon", result);
    }

    return result;
}

template<typename T, typename HardwareTarget>
void EpsilonGreedy<T, HardwareTarget>::reset() {
    // La réinitialisation est implicite car les paramètres sont constants
    // Mais on pourrait ajouter d'autres états internes si nécessaire
}

template<typename T, typename HardwareTarget>
void EpsilonGreedy<T, HardwareTarget>::adaptToMetrics(
    const typename ExplorationStrategy<T, HardwareTarget>::metrics_type& metrics
) {
    // Adaptation dynamique basée sur l'efficacité d'exploration
    T explorationEfficiency = metrics.getExplorationEfficiency();
    T learningProgress = metrics.getLearningProgress();

    // Ajustement dynamique du taux de décroissance
    if (learningProgress > static_cast<T>(0.8)) {
        // Accélérer la décroissance quand l'apprentissage progresse bien
        decay = std::max(decay * static_cast<T>(0.95), static_cast<T>(0.9));
    }
    else if (explorationEfficiency < static_cast<T>(0.3)) {
        // Ralentir la décroissance quand l'exploration est inefficace
        decay = std::min(decay * static_cast<T>(1.05), static_cast<T>(0.999));
    }
}

template<typename T, typename HardwareTarget>
std::unique_ptr<ExplorationStrategy<T, HardwareTarget>> EpsilonGreedy<T, HardwareTarget>::clone() const {
    return std::make_unique<EpsilonGreedy<T, HardwareTarget>>(epsilonStart, epsilonEnd, decay, minSteps);
}

#ifdef USE_GPU
template<typename T, typename HardwareTarget>
__device__ T EpsilonGreedy<T, HardwareTarget>::getExplorationRateDevice(int64_t step) const {
    if (step < minSteps) {
        return epsilonStart;
    }
    else {
        return epsilonEnd + (epsilonStart - epsilonEnd) *
            expf(-1.0f * static_cast<float>(step - minSteps) / (decay * static_cast<float>(minSteps)));
    }
}
#endif

// Explicitement instancier les templates pour les types courants
template class EpsilonGreedy<float, CPU>;
template class EpsilonGreedy<double, CPU>;
#ifdef USE_GPU
template class EpsilonGreedy<float, GPU>;
template class EpsilonGreedy<double, GPU>;
#endif