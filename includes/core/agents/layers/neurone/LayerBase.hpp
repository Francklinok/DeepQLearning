#include <iostream>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "general_concepts.hpp"
#include "neuron_concept.hpp"
#include "tensort_strutur.hpp"

namespace Layer {
    using namespace Layer::Tensor;
    using namespace Layer::Concept;
    using namespace General::Concept;

    // Base class
    template <FloatingPoint T>
    class BaseLayer {
    public:
        using input_type = Tensor<T>;
        using output_type = Tensor<T>;
        using param_type = Tensor<T>;

        virtual ~BaseLayer() = default;

        // Forward / Backward
        virtual output_type forward(const input_type& input) = 0;
        virtual input_type backward(const output_type& output) = 0;

        // Parameters
        virtual std::vector<param_type> get_params() const = 0;
        virtual void set_params(const std::vector<param_type>& params) = 0;

        // Layer name
        virtual std::string_view name() const = 0;

        // Training mode
        virtual void set_training(bool training) {
            training_ = training;
        }

        bool is_training() const {
            return training_;
        }

    protected:
        bool training_ = true;
    };
}
