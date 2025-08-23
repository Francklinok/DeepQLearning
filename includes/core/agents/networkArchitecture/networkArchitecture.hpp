#include <iostream>
#include <tensorflow/cc/framework/ops.h>
#include <tensorflow/cc/framework/scope.h>
#include <tensorflow/cc/ops/standard_ops.h>

class NetworkArchitecture {
public:
    virtual ~NetworkArchitecture() = default;

    virtual tensorflow::Output buildNetwork(
        tensorflow::Scope& scope,
        tensorflow::Input input,
        int inputSize,
        int outputSize
    ) = 0; // interface pure

    bool noisy = false;
};
