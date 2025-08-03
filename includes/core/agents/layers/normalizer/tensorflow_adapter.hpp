#pragma once

#ifdef TENSORFLOW_ENABLED
#include <tensorflow/cc/framework/ops.h>
#include <tensorflow/cc/framework/scope.h>
#include <tensorflow/cc/ops/standard_ops.h>
#endif

#include "normalization_engine.hpp"
#include <string>
#include <unordered_map>

namespace normalization {

#ifdef TENSORFLOW_ENABLED
    // Adaptateur pour intégrer le système de normalisation avec TensorFlow
    template<FloatingPoint T>
    class TensorFlowNormalizationAdapter {
    private:
        std::unique_ptr<NormalizationEngine<T>> engine;

        // Cache des paramètres TensorFlow pour éviter les recalculs
        mutable std::unordered_map<std::string, tensorflow::Output> parameter_cache;

        // Conversion entre nos types et les types TensorFlow
        tensorflow::DataType get_tf_datatype() const {
            if constexpr (std::same_as<T, float>) {
                return tensorflow::DataType::DT_FLOAT;
            }
            else if constexpr (std::same_as<T, double>) {
                return tensorflow::DataType::DT_DOUBLE;
            }
            else {
                static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,
                    "Only float and double are supported for TensorFlow integration");
            }
        }

    public:
        explicit TensorFlowNormalizationAdapter()
            : engine(create_normalization_engine<T>()) {
        }

        // Batch Normalization optimisée pour TensorFlow
        tensorflow::Output batch_normalize(tensorflow::Scope& scope,
            const tensorflow::Input& input,
            const std::string& layer_name,
            T epsilon = T{ 1e-5 },
            bool training = true) const {

            using namespace tensorflow::ops;

            const std::string scale_name = layer_name + "_scale";
            const std::string offset_name = layer_name + "_offset";
            const std::string mean_name = layer_name + "_mean";
            const std::string variance_name = layer_name + "_variance";

            // Création ou récupération des paramètres depuis le cache
            tensorflow::Output scale, offset, mean, variance;

            if (parameter_cache.find(scale_name) == parameter_cache.end()) {
                // Initialisation des paramètres
                scale = Const(scope.WithOpName(scale_name), T{ 1 },
                    tensorflow::TensorShape{/* dimensions basées sur input */ });
                offset = Const(scope.WithOpName(offset_name), T{ 0 },
                    tensorflow::TensorShape{/* dimensions basées sur input */ });

                parameter_cache[scale_name] = scale;
                parameter_cache[offset_name] = offset;
            }
            else {
                scale = parameter_cache[scale_name];
                offset = parameter_cache[offset_name];
            }

            if (training) {
                // Mode entraînement: calcul des statistiques sur le batch
                auto moments = tensorflow::ops::Moments(scope, input, { 0 });
                mean = moments.mean;
                variance = moments.variance;

                // Utilisation de FusedBatchNorm pour de meilleures performances
                auto bn_result = FusedBatchNorm(scope.WithOpName(layer_name + "_bn"),
                    input, scale, offset, mean, variance,
                    FusedBatchNorm::Epsilon(epsilon)
                    .IsTraining(training));

                return bn_result.y;
            }
            else {
                // Mode inférence: utilisation des moyennes et variances mobiles
                if (parameter_cache.find(mean_name) == parameter_cache.end()) {
                    // Initialisation des statistiques mobiles
                    mean = Const(scope.WithOpName(mean_name), T{ 0 },
                        tensorflow::TensorShape{/* dimensions */ });
                    variance = Const(scope.WithOpName(variance_name), T{ 1 },
                        tensorflow::TensorShape{/* dimensions */ });

                    parameter_cache[mean_name] = mean;
                    parameter_cache[variance_name] = variance;
                }
                else {
                    mean = parameter_cache[mean_name];
                    variance = parameter_cache[variance_name];
                }

                auto bn_result = FusedBatchNorm(scope.WithOpName(layer_name + "_bn"),
                    input, scale, offset, mean, variance,
                    FusedBatchNorm::Epsilon(epsilon)
                    .IsTraining(false));

                return bn_result.y;
            }
        }

        // Layer Normalization optimisée pour TensorFlow
        tensorflow::Output layer_normalize(tensorflow::Scope& scope,
            const tensorflow::Input& input,
            const std::string& layer_name,
            T epsilon = T{ 1e-5 },
            const std::vector<int>& axes = { -1 }) const {

            using namespace tensorflow::ops;

            // Calcul des moments sur les axes spécifiés
            auto moments = tensorflow::ops::Moments(scope, input, axes,
                tensorflow::ops::Moments::KeepDims(true));

            // Normalisation
            auto normalized = Div(scope.WithOpName(layer_name + "_div"),
                Sub(scope, input, moments.mean),
                Sqrt(scope, Add(scope, moments.variance,
                    Const(scope, epsilon))));

            // Paramètres d'échelle et d'offset optionnels
            const std::string scale_name = layer_name + "_gamma";
            const std::string offset_name = layer_name + "_beta";

            tensorflow::Output scale, offset;

            if (parameter_cache.find(scale_name) == parameter_cache.end()) {
                // Création des paramètres learnable
                auto input_shape = Shape(scope, input);
                auto feature_size = Slice(scope, input_shape,
                    Const(scope, std::vector<int>{-1}),
                    Const(scope, std::vector<int>{1}));

                scale = Const(scope.WithOpName(scale_name), T{ 1 }, feature_size);
                offset = Const(scope.WithOpName(offset_name), T{ 0 }, feature_size);

                parameter_cache[scale_name] = scale;
                parameter_cache[offset_name] = offset;
            }
            else {
                scale = parameter_cache[scale_name];
                offset = parameter_cache[offset_name];
            }

            // Application de la transformation affine
            return Add(scope.WithOpName(layer_name + "_ln"),
                Mul(scope, normalized, scale), offset);
        }

        // Interface générique utilisant notre moteur de normalisation
        tensorflow::Output normalize(tensorflow::Scope& scope,
            const tensorflow::Input& input,
            NormalizationType type,
            const std::string& layer_name,
            T epsilon = T{ 1e-5 }) const {

            switch (type) {
            case NormalizationType::Batch:
                return batch_normalize(scope, input, layer_name, epsilon);

            case NormalizationType::Layer:
                return layer_normalize(scope, input, layer_name, epsilon);

            case NormalizationType::Instance:
                return instance_normalize(scope, input, layer_name, epsilon);

            case NormalizationType::Group:
                return group_normalize(scope, input, layer_name, epsilon);

            default:
                throw std::invalid_argument("Unsupported normalization type for TensorFlow");
            }
        }

        // Instance Normalization pour TensorFlow
        tensorflow::Output instance_normalize(tensorflow::Scope& scope,
            const tensorflow::Input& input,
            const std::string& layer_name,
            T epsilon = T{ 1e-5 }) const {

            using namespace tensorflow::ops;

            // Instance norm = Layer norm appliquée sur les dimensions spatiales
            // Pour un tensor 4D [batch, height, width, channels], on normalise sur [1, 2]
            auto input_rank = Rank(scope, input);
            auto spatial_axes = Range(scope, Const(scope, 1),
                Sub(scope, input_rank, Const(scope, 1)),
                Const(scope, 1));

            auto moments = tensorflow::ops::Moments(scope, input, spatial_axes,
                tensorflow::ops::Moments::KeepDims(true));

            auto normalized = Div(scope.WithOpName(layer_name + "_instance_div"),
                Sub(scope, input, moments.mean),
                Sqrt(scope, Add(scope, moments.variance,
                    Const(scope, epsilon))));

            return normalized;
        }

        // Group Normalization pour TensorFlow
        tensorflow::Output group_normalize(tensorflow::Scope& scope,
            const tensorflow::Input& input,
            const std::string& layer_name,
            T epsilon = T{ 1e-5 },
            int num_groups = 32) const {

            using namespace tensorflow::ops;

            // Récupération de la forme d'entrée
            auto input_shape = Shape(scope, input);
            auto batch_size = Slice(scope, input_shape, Const(scope, std::vector<int>{0}),
                Const(scope, std::vector<int>{1}));
            auto channels = Slice(scope, input_shape, Const(scope, std::vector<int>{-1}),
                Const(scope, std::vector<int>{1}));

            // Vérification que les canaux sont divisibles par le nombre de groupes
            auto channels_per_group = Div(scope, channels, Const(scope, num_groups));

            // Reshape pour grouper les canaux
            auto spatial_dims = Slice(scope, input_shape, Const(scope, std::vector<int>{1}),
                Const(scope, std::vector<int>{-2}));

            std::vector<tensorflow::Output> new_shape_parts = {
                batch_size,
                Const(scope, num_groups),
                channels_per_group
            };

            // Ajout des dimensions spatiales
            for (int i = 1; i < 3; ++i) {  // Supposant des données 4D
                new_shape_parts.push_back(
                    Slice(scope, input_shape, Const(scope, std::vector<int>{i}),
                        Const(scope, std::vector<int>{1}))
                );
            }

            auto new_shape = Concat(scope, new_shape_parts, Const(scope, 0));
            auto reshaped = Reshape(scope, input, new_shape);

            // Calcul des moments par groupe
            std::vector<int> group_axes = { 2, 3, 4 };  // Channels per group + spatial dims
            auto moments = tensorflow::ops::Moments(scope, reshaped, group_axes,
                tensorflow::ops::Moments::KeepDims(true));

            // Normalisation
            auto normalized = Div(scope.WithOpName(layer_name + "_group_div"),
                Sub(scope, reshaped, moments.mean),
                Sqrt(scope, Add(scope, moments.variance,
                    Const(scope, epsilon))));

            // Reshape vers la forme originale
            return Reshape(scope.WithOpName(layer_name + "_group_norm"),
                normalized, input_shape);
        }

        // Fonction utilitaire pour créer des variables TensorFlow
        tensorflow::Output create_variable(tensorflow::Scope& scope,
            const std::string& name,
            const tensorflow::TensorShape& shape,
            T initial_value = T{ 0 }) const {

            using namespace tensorflow::ops;

            auto initializer = Const(scope.WithOpName(name + "_initializer"),
                initial_value, shape);

            return Variable(scope.WithOpName(name), shape, get_tf_datatype());
        }

        // Optimisation: fusion d'opérations pour de meilleures performances
        tensorflow::Output fused_normalize_and_activate(tensorflow::Scope& scope,
            const tensorflow::Input& input,
            NormalizationType norm_type,
            const std::string& activation,
            const std::string& layer_name,
            T epsilon = T{ 1e-5 }) const {

            using namespace tensorflow::ops;

            // Normalisation
            auto normalized = normalize(scope, input, norm_type, layer_name, epsilon);

            // Activation fusionnée
            if (activation == "relu") {
                return Relu(scope.WithOpName(layer_name + "_relu"), normalized);
            }
            else if (activation == "gelu") {
                // GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
                auto x = normalized;
                auto x_cubed = Mul(scope, Mul(scope, x, x), x);
                auto inner = Add(scope, x, Mul(scope, Const(scope, T{ 0.044715 }), x_cubed));
                auto tanh_input = Mul(scope, Const(scope, T{ 0.7978845608 }), inner);  // sqrt(2/π)
                auto tanh_result = Tanh(scope, tanh_input);
                auto gelu = Mul(scope, Const(scope, T{ 0.5 }),
                    Mul(scope, x, Add(scope, Const(scope, T{ 1 }), tanh_result)));
                return gelu;
            }
            else if (activation == "swish") {
                // Swish: x * sigmoid(x)
                return Mul(scope.WithOpName(layer_name + "_swish"),
                    normalized, Sigmoid(scope, normalized));
            }
            else if (activation == "none" || activation.empty()) {
                return normalized;
            }
            else {
                throw std::invalid_argument("Unsupported activation function: " + activation);
            }
        }

        // Interface pour le fine-tuning des hyperparamètres
        void configure_strategy(NormalizationType type, const std::string& config_json) {
            // Parsing de la configuration JSON et application aux stratégies
            // (Implémentation simplifiée)
            engine->reset_performance_stats();
        }

        // Benchmarking intégré
        void benchmark_tensorflow_vs_native(tensorflow::Scope& scope,
            const tensorflow::Input& input,
            NormalizationType type,
            const std::string& layer_name,
            size_t iterations = 100) const {

            std::cout << "\n=== TensorFlow vs Native Benchmark ===\n";
            std::cout << "Normalization Type: " << engine->get_strategy_name(type) << "\n";
            std::cout << "Iterations: " << iterations << "\n\n";

            // Benchmark TensorFlow
            auto start_tf = std::chrono::high_resolution_clock::now();
            for (size_t i = 0; i < iterations; ++i) {
                auto result = normalize(scope, input, type, layer_name + std::to_string(i));
                // Force l'évaluation (dans un vrai cas, on utiliserait une session)
            }
            auto end_tf = std::chrono::high_resolution_clock::now();
            auto tf_duration = std::chrono::duration_cast<std::chrono::microseconds>(end_tf - start_tf);

            std::cout << "TensorFlow Implementation:\n";
            std::cout << "  Total Time: " << tf_duration.count() << " μs\n";
            std::cout << "  Avg per iteration: " << tf_duration.count() / iterations << " μs\n\n";

            // Les statistiques du moteur natif sont déjà disponibles
            const auto& native_stats = engine->get_performance_stats(type);
            if (native_stats.iterations > 0) {
                std::cout << "Native Implementation:\n";
                std::cout << "  Avg per iteration: " << native_stats.avg_time_per_iteration_us << " μs\n";
                std::cout << "  Throughput: " << native_stats.throughput_elements_per_second << " elements/sec\n";

                const double speedup = tf_duration.count() / (iterations * native_stats.avg_time_per_iteration_us);
                std::cout << "  Speedup: " << speedup << "x\n";
            }
        }

        // Nettoyage du cache pour libérer la mémoire
        void clear_parameter_cache() {
            parameter_cache.clear();
        }

        // Accès au moteur natif pour utilisation directe
        NormalizationEngine<T>& get_native_engine() { return *engine; }
        const NormalizationEngine<T>& get_native_engine() const { return *engine; }
    };

    // Factory functions pour les types courants
    std::unique_ptr<TensorFlowNormalizationAdapter<float>> create_tensorflow_adapter_f32() {
        return std::make_unique<TensorFlowNormalizationAdapter<float>>();
    }

    std::unique_ptr<TensorFlowNormalizationAdapter<double>> create_tensorflow_adapter_f64() {
        return std::make_unique<TensorFlowNormalizationAdapter<double>>();
    }

#endif // TENSORFLOW_ENABLED

    // Stub pour la compilation sans TensorFlow
#ifndef TENSORFLOW_ENABLED
    template<FloatingPoint T>
    class TensorFlowNormalizationAdapter {
    public:
        TensorFlowNormalizationAdapter() {
            static_assert(false, "TensorFlow support is not enabled. Define TENSORFLOW_ENABLED to use this class.");
        }
    };
#endif

}