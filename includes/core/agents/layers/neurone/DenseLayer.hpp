#include <iostream>
#include <memory>
#include <string>
#include <string_view>
#include <vector>
#include <stdexcept>
#include <algorithm>

#include "general_concepts.hpp"
#include "neuron_concept.hpp"
#include "tensort_strutur.hpp"
#include "LayerBase.hpp"

namespace Layer {
    using namespace Layer::Tensor;
    using namespace Layer::Concept;
    using namespace General::Concept;

    // ================= Dense Layer =================
    template<FloatingPoint T>
    class Dense : public BaseLayer<T> {
    public:
        using input_type = Tensor<T>;
        using output_type = Tensor<T>;
        using param_type = Tensor<T>;

    private:
        size_t input_size_;
        size_t output_size_;
        param_type weights_;
        param_type bias_;
        std::unique_ptr<ActivationBase<T>> activation_;
        bool use_bias_;

        mutable input_type cached_input_;
        mutable output_type cached_output_;

    public:
        Dense(size_t input_size, size_t output_size,
            std::unique_ptr<ActivationBase<T>> activation = nullptr,
            bool use_bias = true)
            : input_size_(input_size), output_size_(output_size),
            activation_(std::move(activation)), use_bias_(use_bias) {
            initialize_weights();
        }

        output_type forward(const input_type& input) override {
            if (input.size() != input_size_) {
                throw std::invalid_argument("Input size mismatch");
            }
            cached_input_ = input;

            // Linear transformation
            output_type output = input.matmul(weights_);
            if (use_bias_) {
                output += bias_;
            }

            // Apply activation
            if (activation_) {
                activation_->forward_tensor(output);
            }

            cached_output_ = output;
            return output;
        }

        input_type backward(const output_type& grad_output) override {
            // Compute gradients wrt weights and bias
            param_type weight_grad = cached_input_.transpose().matmul(grad_output);
            param_type bias_grad;

            if (use_bias_) {
                bias_grad = grad_output.sum(0); // sum over batch dimension
            }

            // Gradient wrt input
            input_type input_grad = grad_output.matmul(weights_.transpose());
            return input_grad;
        }

        std::vector<param_type> get_params() const override {
            std::vector<param_type> params = { weights_ };
            if (use_bias_) {
                params.push_back(bias_);
            }
            return params;
        }

        void set_params(const std::vector<param_type>& params) override {
            if (params.empty()) return;
            weights_ = params[0];
            if (use_bias_ && params.size() > 1) {
                bias_ = params[1];
            }
        }

        std::string_view name() const override {
            return "DenseLayer";
        }

    private:
        void initialize_weights() {
            weights_ = Tensor<T>({ input_size_, output_size_ });
            weights_.xavier_init(input_size_, output_size_);
            if (use_bias_) {
                bias_ = Tensor<T>({ output_size_ });
                bias_.fill(T{ 0 });
            }
        }
    };

    // ================= Conv2D Layer =================
    template<FloatingPoint T>
    class Conv2D : public BaseLayer<T> {
    public:
        using input_type = Tensor<T>;
        using output_type = Tensor<T>;
        using param_type = Tensor<T>;

    private:
        size_t in_channels_;
        size_t out_channels_;
        size_t kernel_size_;
        size_t stride_;
        size_t padding_;
        param_type kernels_;
        param_type bias_;
        std::unique_ptr<ActivationBase<T>> activation_;
        bool use_bias_;

        mutable input_type cached_input_;

    public:
        Conv2D(size_t in_channels, size_t out_channels, size_t kernel_size,
            size_t stride = 1, size_t padding = 0,
            std::unique_ptr<ActivationBase<T>> activation = nullptr,
            bool use_bias = true)
            : in_channels_(in_channels), out_channels_(out_channels),
            kernel_size_(kernel_size), stride_(stride), padding_(padding),
            activation_(std::move(activation)), use_bias_(use_bias) {
            initialize_weights();
        }

        output_type forward(const input_type& input) override {
            cached_input_ = input;

            // Simplified convolution placeholder
            output_type output = convolve(input, kernels_);

            if (use_bias_) {
                add_bias(output);
            }

            if (activation_) {
                activation_->forward_tensor(output);
            }

            return output;
        }

        input_type backward(const output_type& grad_output) override {
            return grad_output; // Placeholder
        }

        std::vector<param_type> get_params() const override {
            std::vector<param_type> params = { kernels_ };
            if (use_bias_) {
                params.push_back(bias_);
            }
            return params;
        }

        void set_params(const std::vector<param_type>& params) override {
            if (params.empty()) return;
            kernels_ = params[0];
            if (use_bias_ && params.size() > 1) {
                bias_ = params[1];
            }
        }

        std::string_view name() const override {
            return "ConvLayer";
        }

    private:
        void initialize_weights() {
            size_t fan_in = in_channels_ * kernel_size_ * kernel_size_;
            size_t fan_out = out_channels_ * kernel_size_ * kernel_size_;

            kernels_ = Tensor<T>({ out_channels_, in_channels_, kernel_size_, kernel_size_ });
            kernels_.xavier_init(fan_in, fan_out);

            if (use_bias_) {
                bias_ = Tensor<T>({ out_channels_ });
                bias_.fill(T{ 0 });
            }
        }

        output_type convolve(const input_type& input, const param_type& kernels) const {
            return input; // Placeholder
        }

        void add_bias(output_type& output) const {
            // Simplified bias addition
        }
    };

    // ================= LSTMCell =================
    template<FloatingPoint T>
    class LSTMCell : public BaseLayer<T> {
    public:
        using input_type = Tensor<T>;
        using output_type = Tensor<T>;
        using param_type = Tensor<T>;

        struct LSTMState {
            Tensor<T> hidden;
            Tensor<T> cell;
        };

    private:
        size_t input_size_;
        size_t hidden_size_;

        param_type W_input_;
        param_type W_hidden_;
        param_type bias_;

        std::unique_ptr<ActivationBase<T>> sigmoid_;
        std::unique_ptr<ActivationBase<T>> tanh_;

        mutable LSTMState state_;

    public:
        LSTMCell(size_t input_size, size_t hidden_size)
            : input_size_(input_size), hidden_size_(hidden_size) {
            initialize_weights();
            sigmoid_ = std::make_unique<Sigmoid<T>>();
            tanh_ = std::make_unique<Tanh<T>>();

            state_.hidden = Tensor<T>({ hidden_size_ });
            state_.cell = Tensor<T>({ hidden_size_ });
            reset_state();
        }

        output_type forward(const input_type& input) override {
            if (input.size() != input_size_) {
                throw std::invalid_argument("Input size mismatch");
            }

            Tensor<T> combined({ input_size_ + hidden_size_ });
            std::copy_n(input.data(), input_size_, combined.data());
            std::copy_n(state_.hidden.data(), hidden_size_, combined.data() + input_size_);

            Tensor<T> gates = combined.matmul(W_input_) + bias_;

            size_t gate_size = hidden_size_;
            Tensor<T> forget_gate({ gate_size });
            Tensor<T> input_gate({ gate_size });
            Tensor<T> cell_gate({ gate_size });
            Tensor<T> output_gate({ gate_size });

            std::copy_n(gates.data(), gate_size, forget_gate.data());
            std::copy_n(gates.data() + gate_size, gate_size, input_gate.data());
            std::copy_n(gates.data() + 2 * gate_size, gate_size, cell_gate.data());
            std::copy_n(gates.data() + 3 * gate_size, gate_size, output_gate.data());

            sigmoid_->forward_tensor(forget_gate);
            sigmoid_->forward_tensor(input_gate);
            tanh_->forward_tensor(cell_gate);
            sigmoid_->forward_tensor(output_gate);

            state_.cell *= forget_gate;
            Tensor<T> new_cell_content = input_gate;
            new_cell_content *= cell_gate;
            state_.cell += new_cell_content;

            Tensor<T> cell_tanh = state_.cell;
            tanh_->forward_tensor(cell_tanh);
            state_.hidden = output_gate;
            state_.hidden *= cell_tanh;

            return state_.hidden;
        }

        input_type backward(const output_type& grad_output) override {
            return grad_output; // Placeholder
        }

        std::vector<param_type> get_params() const override {
            return { W_input_, W_hidden_, bias_ };
        }

        void set_params(const std::vector<param_type>& params) override {
            if (params.size() >= 3) {
                W_input_ = params[0];
                W_hidden_ = params[1];
                bias_ = params[2];
            }
        }

        std::string_view name() const override {
            return "LSTMCell";
        }

        void reset_state() {
            state_.hidden.fill(T{ 0 });
            state_.cell.fill(T{ 0 });
        }

        const LSTMState& get_state() const { return state_; }
        void set_state(const LSTMState& state) { state_ = state; }

    private:
        void initialize_weights() {
            size_t total_size = input_size_ + hidden_size_;
            W_input_ = Tensor<T>({ total_size, 4 * hidden_size_ });
            W_hidden_ = Tensor<T>({ hidden_size_, 4 * hidden_size_ });
            bias_ = Tensor<T>({ 4 * hidden_size_ });

            W_input_.xavier_init(total_size, 4 * hidden_size_);
            W_hidden_.xavier_init(hidden_size_, 4 * hidden_size_);
            bias_.fill(T{ 0 });
        }
    };

} // namespace Layer
