
#include  <CompositeExploration.h>

template<typename T, typename HardwareTarget>
CompositeExploration<T, HardwareTarget>::CompositeExploration()
    : combinator([](const std::vector<T>& rates) {
    if (rates.empty()) return static_cast<T>(0);
    return std::accumulate(rates.begin(), rates.end(), static_cast<T>(0)) / static_cast<T>(rates.size());
        }) {
}

template<typename T, typename HardwareTarget>
CompositeExploration<T, HardwareTarget>::CompositeExploration(CombinationFunction combFunc)
    : combinator(std::move(combFunc)) {
}

template<typename T, typename HardwareTarget>
void CompositeExploration<T, HardwareTarget>::addStrategy(std::unique_ptr<ExplorationStrategy<T, HardwareTarget>> strategy, T weight) {
    strategies.push_back({ std::move(strategy), weight });
}

template<typename T, typename HardwareTarget>
T CompositeExploration<T, HardwareTarget>::getExplorationRate(int64_t step, const typename ExplorationStrategy<T, HardwareTarget>::context_type* context) const {
    // Si aucune stratégie, retourner 0
    if (strategies.empty()) return static_cast<T>(0);

    // Utiliser le cache si disponible
    if (context) {
        T cachedValue = context->getCachedValue(step, "composite");
        if (!std::isnan(cachedValue)) {
            return cachedValue;
        }
    }

    // Calculer les taux pour chaque stratégie
    std::vector<T> rates;
    rates.reserve(strategies.size());

    T totalWeight = static_cast<T>(0);
    for (const auto& [strategy, weight] : strategies) {
        rates.push_back(strategy->getExplorationRate(step, context) * weight);
        totalWeight += weight;
    }

    // Normaliser les poids si nécessaire
    if (totalWeight > 0 && totalWeight != static_cast<T>(1.0)) {
        for (auto& rate : rates) {
            rate /= totalWeight;
        }
    }

    // Combiner les taux selon la fonction de combinaison
    T result = combinator(rates);

    // Mettre en cache le résultat
    if (context) {
        context->setCachedValue(step, "composite", result);
    }

    return result;
}

template<typename T, typename HardwareTarget>
void CompositeExploration<T, HardwareTarget>::reset() {
    for (auto& [strategy, _] : strategies) {
        strategy->reset();
    }
}

template<typename T, typename HardwareTarget>
void CompositeExploration<T, HardwareTarget>::adaptToMetrics(const typename ExplorationStrategy<T, HardwareTarget>::metrics_type& metrics) {
    for (auto& [strategy, _] : strategies) {
        strategy->adaptToMetrics(metrics);
    }
}

template<typename T, typename HardwareTarget>
std::unique_ptr<ExplorationStrategy<T, HardwareTarget>> CompositeExploration<T, HardwareTarget>::clone() const {
    auto cloned = std::make_unique<CompositeExploration<T, HardwareTarget>>(combinator);

    for (const auto& [strategy, weight] : strategies) {
        cloned->addStrategy(strategy->clone(), weight);
    }

    return cloned;
}