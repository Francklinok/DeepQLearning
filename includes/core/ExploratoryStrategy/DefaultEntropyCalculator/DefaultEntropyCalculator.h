#pragma once

#include "ExecutionContext.h"

/**
 * @brief Calculateur d'entropie par défaut
 */
template<typename T>
class DefaultEntropyCalculator {
public:
    DefaultEntropyCalculator() = default;
    ~DefaultEntropyCalculator() = default;

    /**
     * @brief Calcule l'entropie basée sur le contexte d'exécution
     * @param context Contexte d'exécution contenant des informations d'état
     * @return Valeur d'entropie calculée
     */
    T calculate(const ExecutionContext<T>* context) const;
};