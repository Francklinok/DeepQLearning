#include "DefaultEntropyCalculator.h"

template<typename T>
T DefaultEntropyCalculator<T>::calculate(const ExecutionContext<T>* context) const {
    if (!context) return static_cast<T>(1.0);

    // Utiliser un simple facteur aléatoire comme approximation d'entropie
    return static_cast<T>(0.8) + context->generateNormalRandom() * static_cast<T>(0.1);
}

// Explicitement instancier les templates pour les types courants
template class DefaultEntropyCalculator<float>;
template class DefaultEntropyCalculator<double>;