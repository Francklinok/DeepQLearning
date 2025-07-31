#pragma once

#include "concepts.hpp"
#include <immintrin.h>
#include <span>
#include <algorithm>
#include <cmath>

namespace simd {

    // Détection des capacités SIMD au runtime
    class SimdCapabilities {
    private:
        static bool avx2_supported;
        static bool fma_supported;
        static bool initialized;

        static void detect_features() noexcept;

    public:
        static bool has_avx2() noexcept {
            if (!initialized) detect_features();
            return avx2_supported;
        }

        static bool has_fma() noexcept {
            if (!initialized) detect_features();
            return fma_supported;
        }
    };

    // Utilitaires SIMD pour float
    namespace f32 {
        constexpr size_t VECTOR_SIZE = 8;  // AVX2: 8 floats par vecteur

        // Somme horizontale d'un vecteur AVX2
        inline float horizontal_sum(__m256 vec) noexcept {
            // Somme des 4 paires
            __m128 sum4 = _mm_add_ps(_mm256_extractf128_ps(vec, 1), _mm256_castps256_ps128(vec));
            // Somme des 2 paires restantes
            __m128 sum2 = _mm_add_ps(sum4, _mm_movehl_ps(sum4, sum4));
            // Somme finale
            __m128 sum1 = _mm_add_ss(sum2, _mm_movehdup_ps(sum2));
            return _mm_cvtss_f32(sum1);
        }

        // Calcul vectorisé de la moyenne
        inline float compute_mean_simd(std::span<const float> data) noexcept {
            if (data.size() < VECTOR_SIZE) {
                // Fallback scalaire pour les petits tableaux
                float sum = 0.0f;
                for (float val : data) sum += val;
                return sum / static_cast<float>(data.size());
            }

            __m256 sum_vec = _mm256_setzero_ps();
            const size_t vectorized_size = (data.size() / VECTOR_SIZE) * VECTOR_SIZE;

            // Traitement vectorisé
            for (size_t i = 0; i < vectorized_size; i += VECTOR_SIZE) {
                __m256 data_vec = _mm256_loadu_ps(data.data() + i);
                sum_vec = _mm256_add_ps(sum_vec, data_vec);
            }

            float sum = horizontal_sum(sum_vec);

            // Traitement des éléments restants
            for (size_t i = vectorized_size; i < data.size(); ++i) {
                sum += data[i];
            }

            return sum / static_cast<float>(data.size());
        }

        // Calcul vectorisé de la variance (avec moyenne connue)
        inline float compute_variance_simd(std::span<const float> data, float mean) noexcept {
            if (data.size() < VECTOR_SIZE) {
                // Fallback scalaire
                float sum_sq_diff = 0.0f;
                for (float val : data) {
                    const float diff = val - mean;
                    sum_sq_diff += diff * diff;
                }
                return sum_sq_diff / static_cast<float>(data.size());
            }

            const __m256 mean_vec = _mm256_set1_ps(mean);
            __m256 sum_sq_diff_vec = _mm256_setzero_ps();
            const size_t vectorized_size = (data.size() / VECTOR_SIZE) * VECTOR_SIZE;

            // Traitement vectorisé
            for (size_t i = 0; i < vectorized_size; i += VECTOR_SIZE) {
                __m256 data_vec = _mm256_loadu_ps(data.data() + i);
                __m256 diff_vec = _mm256_sub_ps(data_vec, mean_vec);
                __m256 sq_diff_vec = _mm256_mul_ps(diff_vec, diff_vec);
                sum_sq_diff_vec = _mm256_add_ps(sum_sq_diff_vec, sq_diff_vec);
            }

            float sum_sq_diff = horizontal_sum(sum_sq_diff_vec);

            // Traitement des éléments restants
            for (size_t i = vectorized_size; i < data.size(); ++i) {
                const float diff = data[i] - mean;
                sum_sq_diff += diff * diff;
            }

            return sum_sq_diff / static_cast<float>(data.size());
        }

        // Normalisation vectorisée in-place
        inline void normalize_inplace_simd(std::span<float> data, float mean, float inv_std) noexcept {
            if (data.size() < VECTOR_SIZE) {
                // Fallback scalaire
                for (float& val : data) {
                    val = (val - mean) * inv_std;
                }
                return;
            }

            const __m256 mean_vec = _mm256_set1_ps(mean);
            const __m256 inv_std_vec = _mm256_set1_ps(inv_std);
            const size_t vectorized_size = (data.size() / VECTOR_SIZE) * VECTOR_SIZE;

            // Traitement vectorisé
            for (size_t i = 0; i < vectorized_size; i += VECTOR_SIZE) {
                __m256 data_vec = _mm256_loadu_ps(data.data() + i);
                __m256 diff_vec = _mm256_sub_ps(data_vec, mean_vec);
                __m256 normalized_vec = _mm256_mul_ps(diff_vec, inv_std_vec);
                _mm256_storeu_ps(data.data() + i, normalized_vec);
            }

            // Traitement des éléments restants
            for (size_t i = vectorized_size; i < data.size(); ++i) {
                data[i] = (data[i] - mean) * inv_std;
            }
        }

        // Normalisation avec scale et offset
        inline void normalize_with_affine_simd(std::span<float> data,
            float mean, float inv_std,
            float scale, float offset) noexcept {
            if (data.size() < VECTOR_SIZE) {
                // Fallback scalaire
                for (float& val : data) {
                    val = scale * (val - mean) * inv_std + offset;
                }
                return;
            }

            const __m256 mean_vec = _mm256_set1_ps(mean);
            const __m256 inv_std_vec = _mm256_set1_ps(inv_std);
            const __m256 scale_vec = _mm256_set1_ps(scale);
            const __m256 offset_vec = _mm256_set1_ps(offset);
            const size_t vectorized_size = (data.size() / VECTOR_SIZE) * VECTOR_SIZE;

            // Traitement vectorisé avec FMA si disponible
            for (size_t i = 0; i < vectorized_size; i += VECTOR_SIZE) {
                __m256 data_vec = _mm256_loadu_ps(data.data() + i);
                __m256 diff_vec = _mm256_sub_ps(data_vec, mean_vec);
                __m256 normalized_vec = _mm256_mul_ps(diff_vec, inv_std_vec);

                if (SimdCapabilities::has_fma()) {
                    // Utilise FMA: scale * normalized + offset
                    __m256 result_vec = _mm256_fmadd_ps(scale_vec, normalized_vec, offset_vec);
                    _mm256_storeu_ps(data.data() + i, result_vec);
                }
                else {
                    // Version sans FMA
                    __m256 scaled_vec = _mm256_mul_ps(scale_vec, normalized_vec);
                    __m256 result_vec = _mm256_add_ps(scaled_vec, offset_vec);
                    _mm256_storeu_ps(data.data() + i, result_vec);
                }
            }

            // Traitement des éléments restants
            for (size_t i = vectorized_size; i < data.size(); ++i) {
                data[i] = scale * (data[i] - mean) * inv_std + offset;
            }
        }
    }

    // Utilitaires SIMD pour double (similaire mais avec AVX2 pour doubles)
    namespace f64 {
        constexpr size_t VECTOR_SIZE = 4;  // AVX2: 4 doubles par vecteur

        inline double horizontal_sum(__m256d vec) noexcept {
            __m128d sum2 = _mm_add_pd(_mm256_extractf128_pd(vec, 1), _mm256_castpd256_pd128(vec));
            __m128d sum1 = _mm_add_pd(sum2, _mm_permute_pd(sum2, 1));
            return _mm_cvtsd_f64(sum1);
        }

        // Implémentations similaires à f32 mais pour double...
        // (Code similaire adapté pour double)
    }

} // namespace deep_qn::normalization::v2::simd