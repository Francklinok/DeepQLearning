// ==================== Usage Examples and Utilities ====================
namespace Examples {

    // Example: Simple usage
    template<typename T = float>
    void basic_usage_example() {
        // Create activation functions
        auto relu = Activation::ActivationFactory<T>::template create<Activation::ActivationFactory<T>::Type::ReLU>();
        auto leaky_relu = Activation::ActivationFactory<T>::template create<Activation::ActivationFactory<T>::Type::LeakyReLU>(T(0.01));

        // Test single values
        T x = T(-0.5);
        std::cout << "ReLU(" << x << ") = " << relu->activate(x) << std::endl;
        std::cout << "LeakyReLU(" << x << ") = " << leaky_relu->activate(x) << std::endl;
    }

    // Example: Batch processing
    template<typename T = float>
    void batch_processing_example() {
        auto gelu = Activation::ActivationFactory<T>::template create<Activation::ActivationFactory<T>::Type::GELU>();

        // Prepare data
        std::vector<T> input = { -2.0, -1.0, 0.0, 1.0, 2.0 };
        std::vector<T> output(input.size());

        // Process batch
        gelu->activate_batch(input.data(), output.data(), input.size());

        std::cout << "GELU batch results: ";
        for (T val : output) {
            std::cout << val << " ";
        }
        std::cout << std::endl;
    }

    // Example: Layer usage
    template<typename T = float>
    void layer_usage_example() {
        // Create activation layer
        auto swish_func = Activation::ActivationFactory<T>::template create<Activation::ActivationFactory<T>::Type::Swish>(T(1.0));
        Activation::ActivationLayer<T> layer(std::move(swish_func));

        // Forward pass
        std::vector<T> input = { -1.0, -0.5, 0.0, 0.5, 1.0 };
        std::vector<T> output;
        layer.forward(input, output);

        // Backward pass
        std::vector<T> grad_output(output.size(), T(1.0)); // Assume gradient of 1
        std::vector<T> grad_input;
        layer.backward(input, grad_output, grad_input);

        std::cout << "Layer: " << layer.get_name() << std::endl;
        std::cout << "Input gradients: ";
        for (T val : grad_input) {
            std::cout << val << " ";
        }
        std::cout << std::endl;
    }

    // Example