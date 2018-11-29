#ifndef SGXDNN_MAIN_H
#define SGXDNN_MAIN_H

#include <immintrin.h>
#include <cmath>

extern int p_int;
extern float p;
extern float mid;

extern int p_verif;
extern double inv_p_verif;

extern __m256 p8f;
extern __m256 mid8f;
extern __m256 negmid8f;
extern __m256 zero8f;
extern __m256 inv_shift8f;
extern __m256 six8f;

extern __m128 p4f;
extern __m128 mid4f;
extern __m128 negmid4f;
extern __m128 zero4f;
extern __m128 inv_shift4f;
extern __m128 six4f;

extern "C" {
		void sgxdnn_benchmarks(int num_threads);

		void load_model_float(char* model_json, float** filters);
		void load_model_float_verify(char* model_json, float** filters, bool preproc);

		void predict_float(float* input, float* output, int batch_size);
		void predict_verify_float(float* input, float* output, float** aux_data, int batch_size);

		void slalom_relu(float* inp, float* out, float* blind, int num_elements, char* activation);
		void slalom_relu_single_batch(float* inp, float* out, float* blind, int num_elements, char* activation);
		void slalom_maxpoolrelu(float* inp, float* out, float* blind, long int dim_in[4], long int dim_out[4],
								int window_rows, int window_cols, int row_stride, int col_stride, bool same_padding);

		void slalom_init(bool integrity, bool privacy, int batch_size);
		void slalom_get_r(float* out, int size);
		void slalom_set_z(float* z, float* dest, int size);
		void slalom_blind_input(float* inp, float* out, int size);
		void slalom_blind_input_internal(float* inp, float* out, int size, float* temp);
}

#define REPS 2

// slow modulo operations
inline double mod(double x, int N){
	return fmod(x, static_cast<double>(N));
}

inline double mod_pos(double x, int N){
	return mod(mod(x, N) + N, N);
}

// Macros for fast modulos
#define REDUCE_MOD(lv_x) \
	{lv_x -= floor(lv_x * inv_p_verif) * p_verif;}

#define REDUCE_MOD_TENSOR(lv_tensor) \
	{lv_tensor = lv_tensor - (lv_tensor * inv_p_verif).floor() * static_cast<double>(p_verif);}

// vectorized activations
__m256 inline relu_avx(__m256 z) {
	return _mm256_round_ps(_mm256_mul_ps(_mm256_max_ps(z, zero8f), inv_shift8f), _MM_FROUND_CUR_DIRECTION);
}

__m256 inline relu6_avx(__m256 z) {
	return _mm256_round_ps(_mm256_mul_ps(_mm256_min_ps(_mm256_max_ps(z, zero8f), six8f), inv_shift8f), _MM_FROUND_CUR_DIRECTION);
}

__m256 inline id_avx(__m256 z) {
	return z;
}

// utilities for computing the double-precision dot product of two vectors of floats
inline void extract_two_doubles(__m256 x, __m256d& x0, __m256d& x1) {
	x0 = _mm256_cvtps_pd(_mm256_extractf128_ps(x, 0));
	x1 = _mm256_cvtps_pd(_mm256_extractf128_ps(x, 1));
}

inline void load_two_doubles(float* x, __m256d& x0, __m256d& x1) {
	__m256 xf = _mm256_load_ps(x);
	extract_two_doubles(xf, x0, x1);
}

inline void load_two_doubles(float* x, __m256& xf, __m256d& x0, __m256d& x1) {
	xf = _mm256_load_ps(x);
	extract_two_doubles(xf, x0, x1);
}

inline void load_two_doubles(double* x, __m256d& x0, __m256d& x1) {
	x0 = _mm256_load_pd(x);
	x1 = _mm256_load_pd(x + 4);
}

inline double double_dot_prod(__m256d a0, __m256d a1, __m256d b0, __m256d b1) {
	// pairwise multiplication
	__m256d muls0 = _mm256_mul_pd(a0, b0);
	__m256d muls1 = _mm256_mul_pd(a1, b1);

	// summation tree
	__m256d sum0 = _mm256_hadd_pd(muls0, muls0);
	__m256d sum1 = _mm256_hadd_pd(muls1, muls1);
	return ((double*)&sum0)[0] + ((double*)&sum0)[2] + ((double*)&sum1)[0] + ((double*)&sum1)[2];
}

inline __m256d double_dot_prod_fmadd(__m256d a0, __m256d a1, __m256d b0, __m256d b1, __m256d accu) {
	accu = _mm256_fmadd_pd(a0, b0, accu);
	return _mm256_fmadd_pd(a1, b1, accu);
}

inline double sum_m256d(__m256d x) {
	__m256d s = _mm256_hadd_pd(x, x);
	return ((double*)&s)[0] + ((double*)&s)[2];
}

// parameters for the fused AES + integrity check
typedef struct integrityParams {
	bool integrity;
	bool pointwise_x;
	bool pointwise_z;
	double* res_x;
	double* res_z;
	float* kernel_r_data;
	double* r_left_data;
	double* r_right_data;
	__m256d temp_x[REPS];
	__m256d temp_z[REPS];
} integrityParams;

// empty dummy functions if we don't care about integrity
void inline empty_verif_x(double* res_x, __m256d* temp_x, float* kernel_r_data, __m256 x, int i, int j, int image_size, int ch) {};
void inline empty_verif_z(double* res_z, __m256d* temp_z, double* r_right_data, __m256 z, int i, int j, int ch) {};
void inline empty_verif_z_outer(double* res_z, __m256d* temp_z, double* r_left_data, int i, int image_size) {};

// Adapted from layers/conv2d.hpp
inline void preproc_verif_pointwise_bias(double* res_x, double* bias_r_data, int image_size) {
	for (int i=0; i<image_size; i++) {
		for(int r=0; r<REPS; r++) {
			res_x[i * REPS + r] = bias_r_data[r];
		}
	}
}

inline void preproc_verif_pointwise_X_inner(double* res_x, __m256d* temp_x, float* kernel_r_data, __m256 x, int i, int j, int image_size, int ch_in) {
	__m256d x0, x1, kr0, kr1;
	extract_two_doubles(x, x0, x1);

	for(int r=0; r<REPS; r++) {
		load_two_doubles(kernel_r_data + (r*ch_in + j), kr0, kr1);
		res_x[i * REPS + r] += double_dot_prod(x0, x1, kr0, kr1);
	}

}

inline void preproc_verif_pointwise_Z_inner(double* res_z, __m256d* temp_z, double* r_right_data, __m256 z, int i, int j, int ch_out) {
	__m256d z0, z1, rr0, rr1;
	extract_two_doubles(z, z0, z1);

	for (int r=0; r<REPS; r++) {
		load_two_doubles(r_right_data + (r*ch_out + j), rr0, rr1);
		res_z[i*REPS + r] += double_dot_prod(z0, z1, rr0, rr1);
	}
}

inline void preproc_verif_bias(double* res_x, double* bias_r_data) {
	for (int r=0; r<REPS; r++) {
		res_x[r] = bias_r_data[r];
	}
}

inline void preproc_verif_X_inner(double* res_x, __m256d* temp_x, float* kernel_r_data, __m256 x, int i, int j, int image_size, int ch_in) {
	__m256d x0, x1, kr0, kr1;
	extract_two_doubles(x, x0, x1);

	for (int r=0; r<REPS; r++) {
		load_two_doubles(kernel_r_data + (r * image_size * ch_in + i * ch_in + j), kr0, kr1);
		//res_x[r] += double_dot_prod(x0, x1, kr0, kr1);
		temp_x[r] = double_dot_prod_fmadd(x0, x1, kr0, kr1, temp_x[r]);
	}
}

inline void preproc_verif_Z_inner(double* res_z, __m256d* temp_z, double* r_right_data, __m256 z, int i, int j, int ch_out) {
	__m256d z0, z1, rr0, rr1;
	extract_two_doubles(z, z0, z1);

	for (int r=0; r<REPS; r++) {
		load_two_doubles(r_right_data + (r*ch_out + j), rr0, rr1);
		//res_z_temp[r] += double_dot_prod(z0, z1, rr0, rr1);
		temp_z[r] = double_dot_prod_fmadd(z0, z1, rr0, rr1, temp_z[r]);
	}
}

inline void preproc_verif_Z_outer(double* res_z, __m256d* temp_z, double* r_left_data, int i, int out_image_size) {
	for (int r=0; r<REPS; r++) {
		double rl = r_left_data[r*out_image_size + i];
		//double t = res_z_temp[r];
		double t = sum_m256d(temp_z[r]);
		REDUCE_MOD(t);
		res_z[r] += rl * t;
		//res_z_temp[r] = 0;
		temp_z[r] = _mm256_setzero_pd();
	}
}

#endif
