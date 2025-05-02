#pragma once
#include <iostream>

// Forward declaration of Experience struct to avoid circular dependency
template <typename State, typename Action>
struct Experience;

/**
 * @brief Abstract base class for experience replay buffers.
 *
 * This class defines a common interface for all types of experience replay buffers
 * (e.g., standard, prioritized, persistent), which are used in reinforcement learning
 * to store and retrieve past experiences for training.
 *
 * @tparam State The type representing environment states.
 * @tparam Action The type representing actions taken in the environment.
 */
template <typename State, typename Action>
class AbstractBuffer {
public:
    // Virtual destructor to ensure proper cleanup in derived classes
    virtual ~AbstractBuffer() = default;

    /**
     * @brief Add an experience to the buffer.
     *
     * @param exp The experience to be stored.
     * @param priority The priority of the experience (default is 0.1).
     */
    virtual void add(Experience<State, Action>& exp, float priority = 0.1f) = 0;

    /**
     * @brief Sample a batch of experiences from the buffer.
     *
     * @param batchSize The number of experiences to sample.
     * @return A tuple containing:
     *         - A vector of sampled experiences
     *         - A vector of their corresponding indices in the buffer
     *         - A vector of sampling weights (for prioritized buffers)
     */
    virtual std::tuple<std::vector<Experience<State, Action>>, std::vector<size_t>, std::vector<float>>
        sample(size_t batchSize) = 0;

    /**
     * @brief Update priorities of specific experiences in the buffer.
     *
     * @param indices The indices of the experiences to update.
     * @param newPriorities The new priority values for each corresponding index.
     */
    virtual void updatePriorities(const std::vector<size_t>& indices, const std::vector<float>& newPriorities) = 0;

    /**
     * @brief Get the current size of the buffer (number of stored experiences).
     *
     * @return The number of experiences stored.
     */
    virtual size_t size() const = 0;
};
