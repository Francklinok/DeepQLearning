#pragma once

#include <cmath>
#include "ExecutionContext.h"

/**
 * @brief Calculateur d'entropie par défaut basé sur une approximation aléatoire.
 * @tparam T Type numérique (float, double, etc.)
 */
template<typename T>
class DefaultEntropyCalculator {
public:
    DefaultEntropyCalculator() = default;
    ~DefaultEntropyCalculator() = default;

    /**
     * @brief Calcule l'entropie basée sur le contexte d'exécution.
     * @param context Pointeur vers le contexte d'exécution.
     * @return Valeur d'entropie estimée (entre 0.7 et 0.9 environ).
     */
    inline T calculate(const ExecutionContext<T>* context) const {
        if (!context) return static_cast<T>(1.0);
        // Utiliser un simple facteur aléatoire comme approximation d'entropie
        return static_cast<T>(0.8) + context->generateNormalRandom() * static_cast<T>(0.1);
    }
};

// Instanciations explicites pour les types courants
template class DefaultEntropyCalculator<float>;
template class DefaultEntropyCalculator<double>;
