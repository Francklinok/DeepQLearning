// Implémentation inline des méthodes statiques de la classe ExplorationFactory

template<typename T, typename HardwareTarget>
std::unique_ptr<ExplorationStrategy<T, HardwareTarget>> ExplorationFactory<T, HardwareTarget>::createEpsilonGreedy(
    T start, T end, T decay, int64_t minSteps
) {
    return std::make_unique<EpsilonGreedy<T, HardwareTarget>>(start, end, decay, minSteps);
}

template<typename T, typename HardwareTarget>
std::unique_ptr<ExplorationStrategy<T, HardwareTarget>> ExplorationFactory<T, HardwareTarget>::createNoisyExploration(
    T scale, T decay, bool adaptiveNoise
) {
    return std::make_unique<NoisyExploration<T, HardwareTarget>>(scale, decay, adaptiveNoise);
}

template<typename T, typename HardwareTarget>
std::unique_ptr<ExplorationStrategy<T, HardwareTarget>> ExplorationFactory<T, HardwareTarget>::createEntropyBasedExploration(
    T weight, T minWeight, T decay, T adaptiveFactor
) {
    return std::make_unique<EntropyBasedExploration<T, HardwareTarget>>(
        weight, minWeight, decay, adaptiveFactor);
}

template<typename T, typename HardwareTarget>
std::unique_ptr<ExplorationStrategy<T, HardwareTarget>> ExplorationFactory<T, HardwareTarget>::createBoltzmannExploration(
    T initialTemp, T minTemp, T coolingRate, bool adaptiveCooling
) {
    return std::make_unique<BoltzmannExploration<T, HardwareTarget>>(
        initialTemp, minTemp, coolingRate, adaptiveCooling);
}

template<typename T, typename HardwareTarget>
std::unique_ptr<ExplorationStrategy<T, HardwareTarget>> ExplorationFactory<T, HardwareTarget>::createBalancedComposite() {
    auto composite = std::make_unique<CompositeExploration<T, HardwareTarget>>();

    composite->addStrategy(createEpsilonGreedy(), static_cast<T>(1.0));
    composite->addStrategy(createBoltzmannExploration(), static_cast<T>(1.0));
    composite->addStrategy(createNoisyExploration(), static_cast<T>(0.5));

    return composite;
}

template<typename T, typename HardwareTarget>
std::unique_ptr<ExplorationStrategy<T, HardwareTarget>> ExplorationFactory<T, HardwareTarget>::createAdaptiveComposite() {
    // Fonction de combinaison qui favorise les valeurs extrêmes
    auto nonLinearCombinator = [](const std::vector<T>& rates) {
        if (rates.empty()) return static_cast<T>(0);

        // Calculer le produit des compléments et soustraire du 1
        T product = static_cast<T>(1.0);
        for (T rate : rates) {
            product *= (static_cast<T>(1.0) - rate);
        }

        return static_cast<T>(1.0) - product;
        };

    auto composite = std::make_unique<CompositeExploration<T, HardwareTarget>>(nonLinearCombinator);

    composite->addStrategy(createEpsilonGreedy(static_cast<T>(0.8), static_cast<T>(0.01), static_cast<T>(0.998), 2000), static_cast<T>(1.2));
    composite->addStrategy(createBoltzmannExploration(static_cast<T>(0.5), static_cast<T>(0.01), static_cast<T>(0.997), true), static_cast<T>(1.0));
    composite->addStrategy(createEntropyBasedExploration(), static_cast<T>(0.8));

    return composite;
}