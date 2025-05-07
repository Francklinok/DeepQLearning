#include "EntropyBasedExploration.h"
#include <cmath>
#include <algorithm>

template<
    typename T,
    typename HardwareTarget,
    template<typename> class EntropyCalculator
>
EntropyBasedExploration<T, HardwareTarget, EntropyCalculator>::EntropyBasedExploration(
    T weight,
    T minWeight,
    T decay,
    T adaptiveFactor
)
    : entropyWeight(weight)
    , minEntropyWeight(minWeight)
    , decay(decay)
    , initialWeight(weight)
    , adaptiveFactor(adaptiveFactor)
    , entropyCalculator()
{
    // Validation des paramètres
    if (entropyWeight < minEntropyWeight) {
        std::swap(entropyWeight, minEntropyWeight);
    }

    if (decay <= 0 || decay >= 1) {
        decay = static_cast<T>(0.995);
    }

    if (adaptiveFactor <= 1) {
        adaptiveFactor = static_cast<T>(1.2);
    }
}

template<
    typename T,
    typename HardwareTarget,
    template<typename> class EntropyCalculator
>
T EntropyBasedExploration<T, HardwareTarget, EntropyCalculator>::getExplorationRate(
    int64_t step,
    const typename ExplorationStrategy<T, HardwareTarget>::context_type* context
) const {
    // Utiliser le cache si disponible
    if (context) {
        T cachedValue = context->getCachedValue(step, "entropy");
        if (!std::isnan(cachedValue)) {
            return cachedValue;
        }
    }

    // Calcul de base avec décroissance exponentielle
    T baseRate = minEntropyWeight + (entropyWeight - minEntropyWeight) *
        std::exp(-static_cast<T>(0.001) * static_cast<T>(step));

    // Correction de l'entropie si le contexte est disponible
    if (context) {
        T entropyFactor = entropyCalculator.calculate(context);
        baseRate *= std::pow(entropyFactor, adaptiveFactor);
    }

    // Limiter le taux entre 0 et 1
    T result = std::clamp(baseRate, static_cast<T>(0), static_cast<T>(1));

    // Mettre en cache le résultat
    if (context) {
        context->setCachedValue(step, "entropy", result);
    }

    return result;
}

template<
    typename T,
    typename HardwareTarget,
    template<typename> class EntropyCalculator
>
void EntropyBasedExploration<T, HardwareTarget, EntropyCalculator>::reset() {
    entropyWeight = initialWeight;
}

template<
    typename T,
    typename HardwareTarget,
    template<typename> class EntropyCalculator
>
void EntropyBasedExploration<T, HardwareTarget, EntropyCalculator>::adaptToMetrics(
    const typename ExplorationStrategy<T, HardwareTarget>::metrics_type& metrics
) {
    T entropyLevel = metrics.getEntropyLevel();
    T learningProgress = metrics.getLearningProgress();

    // Ajustement dynamique basé sur l'entropie et le progrès d'apprentissage
    if (entropyLevel < static_cast<T>(0.2) && learningProgress < static_cast<T>(0.5)) {
        // Augmenter le poids de l'entropie si l'entropie est faible et l'apprentissage lent
        entropyWeight = std::min(entropyWeight * adaptiveFactor, initialWeight * static_cast<T>(3.0));
    }
    else if (entropyLevel > static_cast<T>(0.8) || learningProgress > static_cast<T>(0.8)) {
        // Réduire le poids si l'entropie est élevée ou l'apprentissage avancé
        entropyWeight = std::max(entropyWeight / adaptiveFactor, minEntropyWeight);
    }
}

template<
    typename T,
    typename HardwareTarget,
    template<typename> class EntropyCalculator
>
std::unique_ptr<ExplorationStrategy<T, HardwareTarget>>
EntropyBasedExploration<T, HardwareTarget, EntropyCalculator>::clone() const {
    return std::make_unique<EntropyBasedExploration<T, HardwareTarget, EntropyCalculator>>(
        entropyWeight, minEntropyWeight, decay, adaptiveFactor);
}

#ifdef USE_GPU
template<
    typename T,
    typename HardwareTarget,
    template<typename> class EntropyCalculator
>
__device__ T EntropyBasedExploration<T, HardwareTarget, EntropyCalculator>::getExplorationRateDevice(int64_t step) const {
    T baseRate = minEntropyWeight + (entropyWeight - minEntropyWeight) *
        expf(-0.001f * static_cast<float>(step));

    // Limiter entre 0 et 1
    return max(0.0f, min(1.0f, baseRate));
}
#endif

// Explicitement instancier les templates pour les types courants
template class EntropyBasedExploration<float, CPU, DefaultEntropyCalculator>;
template class EntropyBasedExploration<double, CPU, DefaultEntropyCalculator>;

#ifdef USE_GPU
template class EntropyBasedExploration<float, GPU, DefaultEntropyCalculator>;
template class EntropyBasedExploration<double, GPU, DefaultEntropyCalculator>;
#endif