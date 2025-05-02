#pragma once
#include "abstractBuffer/abstractBuffer.h"

/**
 * @brief Adapter class to wrap any buffer implementation with a common interface.
 *
 * This class allows different types of buffer implementations (such as prioritized buffers)
 * to be used interchangeably through the AbstractBuffer interface.
 *
 * @tparam State The type representing environment states.
 * @tparam Action The type representing actions taken in the environment.
 * @tparam BufferImpl The concrete buffer implementation to adapt.
 */
template <typename State, typename Action, typename BufferImpl>
class BufferAdapter : public AbstractBuffer<State, Action> {
private:
    // The actual buffer implementation (e.g., PrioritizedBuffer, RegularBuffer, etc.)
    BufferImpl buffer;

public:
    /**
     * @brief Constructor that perfectly forwards arguments to the buffer implementation.
     *
     * @tparam Args Types of constructor arguments.
     * @param args Arguments to initialize the internal buffer.
     */
    template <typename... Args>
    BufferAdapter(Args&&... args);

    /**
     * @brief Add an experience to the buffer with an optional priority.
     *
     * @param exp The experience to add.
     * @param priority Priority value used by prioritized buffers (default is 0.1).
     */
    void add(Experience<State, Action>& exp, float priority = 0.1f) override;

    /**
     * @brief Sample a batch of experiences from the buffer.
     *
     * @param batchSize Number of experiences to sample.
     * @return A tuple containing:
     *         - A vector of sampled experiences.
     *         - A vector of corresponding indices.
     *         - A vector of sampling weights.
     */
    std::tuple<std::vector<Experience<State, Action>>, std::vector<size_t>, std::vector<float>>
        sample(size_t batchSize) override;

    /**
     * @brief Update the priorities of specific experiences in the buffer.
     *
     * @param indices Indices of the experiences to update.
     * @param newPriorities New priority values.
     */
    void updatePriorities(const std::vector<size_t>& indices, const std::vector<float>& newPriorities) override;

    /**
     * @brief Get the number of experiences currently stored in the buffer.
     *
     * @return The size of the buffer.
     */
    size_t size() const override;
};
