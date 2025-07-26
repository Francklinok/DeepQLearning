#pragma once

#include <cmath>
#include "ExecutionContext.h"

/**
 * @brief Calculateur d'entropie par d�faut bas� sur une approximation al�atoire.
 * @tparam T Type num�rique (float, double, etc.)
 */
template<typename T>
class DefaultEntropyCalculator {
public:
    DefaultEntropyCalculator() = default;
    ~DefaultEntropyCalculator() = default;

    /**
     * @brief Calcule l'entropie bas�e sur le contexte d'ex�cution.
     * @param context Pointeur vers le contexte d'ex�cution.
     * @return Valeur d'entropie estim�e (entre 0.7 et 0.9 environ).
     */
    inline T calculate(const ExecutionContext<T>* context) const {
        if (!context) return static_cast<T>(1.0);
        // Utiliser un simple facteur al�atoire comme approximation d'entropie
        return static_cast<T>(0.8) + context->generateNormalRandom() * static_cast<T>(0.1);
    }
};

// Instanciations explicites pour les types courants
template class DefaultEntropyCalculator<float>;
template class DefaultEntropyCalculator<double>;
