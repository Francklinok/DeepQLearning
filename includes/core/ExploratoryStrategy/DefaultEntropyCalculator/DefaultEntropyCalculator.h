#pragma once

#include "ExecutionContext.h"

/**
 * @brief Calculateur d'entropie par d�faut
 */
template<typename T>
class DefaultEntropyCalculator {
public:
    DefaultEntropyCalculator() = default;
    ~DefaultEntropyCalculator() = default;

    /**
     * @brief Calcule l'entropie bas�e sur le contexte d'ex�cution
     * @param context Contexte d'ex�cution contenant des informations d'�tat
     * @return Valeur d'entropie calcul�e
     */
    T calculate(const ExecutionContext<T>* context) const;
};