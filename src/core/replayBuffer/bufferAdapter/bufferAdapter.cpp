#include "bufferAdapter.h"

// Constructor implementation that perfectly forwards the arguments to the internal buffer implementation.
template <typename State, typename Action, typename BufferImpl>
template <typename... Args>
BufferAdapter<State, Action, BufferImpl>::BufferAdapter(Args&&... args)
	: buffer(std::forward<Args>(args)...) {
}

// Adds an experience to the buffer with a given priority.
template <typename State, typename Action, typename BufferImpl>
void BufferAdapter<State, Action, BufferImpl>::add(Experience<State, Action>& exp, float priority) {
	buffer.add(exp, priority);
}

// Samples a batch of experiences from the underlying buffer.
// Returns a tuple containing the sampled experiences, their indices, and importance-sampling weights.
template <typename State, typename Action, typename BufferImpl>
std::tuple<std::vector<Experience<State, Action>>, std::vector<size_t>, std::vector<float>>
BufferAdapter<State, Action, BufferImpl>::sample(size_t batchSize) {
	return buffer.sample(batchSize);
}

// Updates the priorities of specific experiences using their indices.
template <typename State, typename Action, typename BufferImpl>
void BufferAdapter<State, Action, BufferImpl>::updatePriorities(const std::vector<size_t>& indices, const std::vector<float>& newPriorities) {
	buffer.updatePriorities(indices, newPriorities);
}

// Returns the number of experiences currently stored in the buffer.
template <typename State, typename Action, typename BufferImpl>
size_t BufferAdapter<State, Action, BufferImpl>::size() const {
	return buffer.size();
}
