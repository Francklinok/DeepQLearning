#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <chrono>
#include <memory>

#include "ExplorationStrategy.hpp"
#include "EpsilonGreedy.hpp"
#include "NoisyExploration.hpp"
#include "EntropyBasedExploration.hpp"
#include "BoltzmannExploration.hpp"
#include "CompositeExploration.hpp"
#include "ExplorationFactory.hpp"

// Fonction utilitaire pour afficher les taux d'exploration
template<typename T>
void displayExplorationRates(const ExplorationStrategy<T>& strategy, int numSteps) {
    std::cout << std::fixed << std::setprecision(4);

    auto context = std::make_unique<ExecutionContext<T>>(true);

    for (int step = 0; step <= numSteps; step += numSteps / 10) {
        T rate = strategy.getExplorationRate(step, context.get());
        std::cout << "�tape " << std::setw(6) << step << ": taux = " << rate << std::endl;
    }
    std::cout << std::endl;
}

// Exemple d'utilisation des diff�rentes strat�gies
int main() {
    std::cout << "=== D�monstration des strat�gies d'exploration ===" << std::endl << std::endl;

    // Param�tres de test
    const int numSteps = 10000;

    // 1. Strat�gie Epsilon-Greedy
    std::cout << "1. Strat�gie Epsilon-Greedy:" << std::endl;
    auto epsilonGreedy = ExplorationFactory<float>::createEpsilonGreedy(1.0f, 0.01f, 0.995f, 1000);
    displayExplorationRates(*epsilonGreedy, numSteps);

    // 2. Strat�gie Boltzmann
    std::cout << "2. Strat�gie Boltzmann:" << std::endl;
    auto boltzmann = ExplorationFactory<float>::createBoltzmannExploration(1.0f, 0.05f, 0.998f, true);
    displayExplorationRates(*boltzmann, numSteps);

    // 3. Strat�gie bas�e sur le bruit
    std::cout << "3. Strat�gie bas�e sur le bruit:" << std::endl;
    auto noisy = ExplorationFactory<float>::createNoisyExploration(0.5f, 0.99f, true);
    displayExplorationRates(*noisy, numSteps);

    // 4. Strat�gie bas�e sur l'entropie
    std::cout << "4. Strat�gie bas�e sur l'entropie:" << std::endl;
    auto entropy = ExplorationFactory<float>::createEntropyBasedExploration(0.01f, 0.001f, 0.995f, 1.2f);
    displayExplorationRates(*entropy, numSteps);

    // 5. Strat�gie composite �quilibr�e
    std::cout << "5. Strat�gie composite �quilibr�e:" << std::endl;
    auto balanced = ExplorationFactory<float>::createBalancedComposite();
    displayExplorationRates(*balanced, numSteps);

    // 6. Strat�gie composite adaptative
    std::cout << "6. Strat�gie composite adaptative:" << std::endl;
    auto adaptive = ExplorationFactory<float>::createAdaptiveComposite();
    displayExplorationRates(*adaptive, numSteps);

    // 7. D�monstration d'adaptation aux m�triques
    std::cout << "7. D�monstration d'adaptation aux m�triques:" << std::endl;

    auto metrics = ExplorationMetrics<float>();
    metrics.setExplorationEfficiency(0.2f);  // Efficacit� d'exploration faible
    metrics.setEntropyLevel(0.1f);           // Entropie faible
    metrics.setLearningProgress(0.8f);       // Progr�s d'apprentissage �lev�

    std::cout << "Avant adaptation:" << std::endl;
    displayExplorationRates(*adaptive, numSteps);

    adaptive->adaptToMetrics(metrics);

    std::cout << "Apr�s adaptation:" << std::endl;
    displayExplorationRates(*adaptive, numSteps);

    // 8. Test de performances
    std::cout << "8. Test de performances:" << std::endl;

    const int perfTestSteps = 1000000;
    std::vector<int64_t> steps(perfTestSteps);
    for (int i = 0; i < perfTestSteps; ++i) {
        steps[i] = i;
    }

    auto startTime = std::chrono::high_resolution_clock::now();

    auto results = adaptive->batchProcess(steps);

    auto endTime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);

    std::cout << "Traitement de " << perfTestSteps << " �tapes en " << duration.count()
        << " ms (" << static_cast<double>(perfTestSteps) / duration.count() * 1000.0
        << " �tapes/seconde)" << std::endl;

    return 0;
}