#ifndef SGXDNN_BENCHMARKS_H_
#define SGXDNN_BENCHMARKS_H_

#include "utils.hpp"
#include "tensor_types.h"
#include "mempool.hpp"
#include "layers/conv2d.hpp"
#include "layers/eigen_spatial_convolutions.h"
#include "layers/depthwise_conv2d.hpp"

#include <stdio.h>
#include <iostream>
#include <string>
#include <type_traits>
#include <assert.h>
#include <cmath>
#include "immintrin.h"

#ifndef USE_SGX
#include "omp.h"
#endif

using namespace tensorflow;
using namespace SGXDNN;

float* X_data;
float* W_data;
float* Z_data;

double* r_left_data;
double* r_right_data;
float* W_r_data;


void conv2d_compute(int batch, int h, int w, int ch_in, int ch_out, void* device_ptr) {

#ifdef EIGEN_USE_THREADS
	Eigen::ThreadPoolDevice* d = static_cast<Eigen::ThreadPoolDevice*>(device_ptr);
#endif

	TensorMap<float, 4> X(X_data, batch, h, w, ch_in);
	TensorMap<float, 4> W(W_data, 3, 3, ch_in, ch_out);
	TensorMap<float, 4> Z(Z_data, batch, h, w, ch_out);

#ifdef EIGEN_USE_THREADS
	Z.device(*d) = Eigen::SpatialConvolution(X, W, 1, 1, Eigen::PaddingType::PADDING_SAME);
#else
	Z = Eigen::SpatialConvolution(X, W, 1, 1, Eigen::PaddingType::PADDING_SAME);
#endif
}

void conv2d_verif_batch(int batch, int h, int w, int ch_in, int ch_out, void* device_ptr, int k_h=3, int k_w=3) {

#ifdef EIGEN_USE_THREADS
	Eigen::ThreadPoolDevice* d = static_cast<Eigen::ThreadPoolDevice*>(device_ptr);
#endif

	Tensor<double, 1> temp(REPS * h * w * ch_out);
	temp.setZero();

	// compute (r_b * X) mod p
	for (int i=0; i<batch; i++) {
		for(int j=0; j<h*w*ch_in; j++) {
			for(int r=0; r<REPS; r++) {
				temp.data()[r*h*w*ch_in + j] += static_cast<double>(X_data[i*h*w*ch_in + j]) * r_left_data[r*batch + i];
			}
		}
	}

	for (int r=0; r<REPS; r++) {
		for (int j=0; j<h*w*ch_in; j++) {
			REDUCE_MOD(temp.data()[r*h*w*ch_in + j]);
		}
	}

	// compute W*r_ch
	TensorMap<float, 4> W(W_data, k_h, k_w, ch_in, ch_out);
	array2d W_dims2d{{k_h * k_w * ch_in, ch_out}};
	array4d W_dims4d{{k_h, k_w, ch_in, 1}};

	// Compute Conv(r_b * X, W*r_ch)
	Tensor<double, 1> out1(REPS * h * w);
	for (int i=0; i<REPS; i++) {
		TensorMap<double, 2> r_ch_temp(r_right_data + i * ch_out, ch_out, 1);
		Tensor<double, 4> W_r_temp = W.reshape(W_dims2d).template cast<double>().contract(r_ch_temp, MATRIX_PRODUCT).reshape(W_dims4d);
		TensorMap<double, 4> X_sum_temp(temp.data() + i*h*w*ch_in, 1, h, w, ch_in);
		TensorMap<double, 4> out_temp(out1.data() + i*h*w, 1, h, w, 1);
		out_temp = Eigen::SpatialConvolution(X_sum_temp, W_r_temp, 1, 1, Eigen::PaddingType::PADDING_SAME);
	}

	temp.setZero();

	// compute r_b * Z
	for (int i=0; i<batch; i++) {
		for(int j=0; j<h*w*ch_out; j++) {
			for(int r=0; r<REPS; r++) {
				temp.data()[r*h*w*ch_out + j] += static_cast<double>(Z_data[i*h*w*ch_out+ j]) * r_left_data[r*batch + i];
			}
		}
	}

	// temp is of shape (REPS, h_out, w_out, ch_out)
	// we have r_b * Z, and we want r_b*Z*r_ch
	for (int r=0; r<REPS; r++) {
		for (int j=0; j<h*w; j++) {
			for (int k=0; k<ch_out; k++) {
				double Z = temp.data()[r*h*w*ch_out + j*ch_out + k];
				REDUCE_MOD(Z);
				out1.data()[r*h*w + j] -= Z * r_right_data[r*ch_out + k];
			}
		}
	}
}

void conv2d_verif_nopreproc(int batch, int h, int w, int ch_in, int ch_out, MemPool* mem_pool, void* device_ptr) {

	#ifdef EIGEN_USE_THREADS
	Eigen::ThreadPoolDevice* d = static_cast<Eigen::ThreadPoolDevice*>(device_ptr);
	#endif

	array2d patch_dims2d{{batch * h * w, 3 * 3 * ch_in}};
	array2d Z_dims2d{{batch * h * w, ch_out}};
	array2d Z_dims_out{{batch * h * w, REPS}};
	array2d W_dims2d{{3 * 3 * ch_in, ch_out}};
	array4d W_dims_out{{3, 3, ch_in, REPS}};

	sgx_time_t start = get_time();
	// compute W*r
	TensorMap<float, 4> W(W_data, 3, 3, ch_in, ch_out);
	TensorMap<double, 2> r(r_right_data, ch_out, REPS);
	#ifdef EIGEN_USE_THREADS
	Tensor<double, 2> W_r(3*3*ch_in, REPS);
	W_r.device(*d) = W.reshape(W_dims2d).template cast<double>().contract(r, MATRIX_PRODUCT);
	#else
	Tensor<double, 2> W_r = W.reshape(W_dims2d).template cast<double>().contract(r, MATRIX_PRODUCT);
	#endif

	// allocate a temp buffer to store Conv(X, W*r)
	Tensor<double, 2> out1(batch * h * w, REPS);

	conv2d_im2col(X_data, batch, h, w, ch_in,
	   W_r.data(), 3, 3, REPS,
	   1, 1, Eigen::PaddingType::PADDING_SAME,
	   out1.data(), h, w, mem_pool);

	// compute Z*r
	TensorMap<float, 4> Z(Z_data, batch, h, w, ch_out);
	#ifdef EIGEN_USE_THREADS
	out1.device(*d) -= Z.reshape(Z_dims2d).template cast<double>().contract(r, MATRIX_PRODUCT);
	#else
	out1 -= Z.reshape(Z_dims2d).template cast<double>().contract(r, MATRIX_PRODUCT);
	#endif

	W_r.resize(0,0);
	out1.resize(0,0);
}

void conv2d_verif_preproc(int batch, int h, int w, int ch_in, int ch_out, void* device_ptr) {

	#ifdef EIGEN_USE_THREADS
	Eigen::ThreadPoolDevice* d = static_cast<Eigen::ThreadPoolDevice*>(device_ptr);
	#endif

	int image_size = h*w;

	Tensor<double, 1> res1_full(REPS);
	Tensor<double, 1> res2_full(REPS);

	#pragma omp parallel
	{

		__m256d res1_private[REPS];
		__m256d res2_private[REPS];

		__m256d x0, x1, kr0, kr1;
		#pragma omp for private(x0, x1, kr0, kr1)
		for (int k=0; k<image_size * ch_in; k+=8) {
			load_two_doubles(X_data + k, x0, x1);

			for (int r=0; r<REPS; r++) {
				load_two_doubles(W_r_data + (r * image_size * ch_in + k), kr0, kr1);
				res1_private[r] = double_dot_prod_fmadd(x0, x1, kr0, kr1, res1_private[r]);
			}
		}

		__m256d z0, z1, rr0, rr1;
		#pragma omp for private(z0, z1, rr0, rr1)
		for (int i=0; i<image_size; i++) {
			__m256d temp[REPS];

			for (int j=0; j<ch_out; j+=8) {
				load_two_doubles(Z_data + (i*ch_out + j), z0, z1);
				for (int r=0; r<REPS; r++) {
					load_two_doubles(r_right_data + (r*ch_out + j), rr0, rr1);
					temp[r] = double_dot_prod_fmadd(z0, z1, rr0, rr1, temp[r]);
				}
			}

			for (int r=0; r<REPS; r++) {
				double rl = r_left_data[r*image_size + i];
				double t = sum_m256d(temp[r]);
				REDUCE_MOD(t);
				res2_private[r] += rl * t;
			}
		}


		#pragma omp critical
		{
			for (int i=0; i<REPS; i++) {
				res1_full.data()[i] += sum_m256d(res1_private[i]);
				res2_full.data()[i] += sum_m256d(res2_private[i]);
			}
		}
	}

	res1_full.resize(0);
	res2_full.resize(0);
}

void benchmark_conv(MemPool* mem_pool, void* device_ptr) {

	int num_conv_benchamrks = 5;
	const unsigned int conv_sizes[num_conv_benchamrks][4] = {
			{224, 224, 64, 64},
			{112, 112, 128, 128},
			{56, 56, 256, 256},
			{28, 28, 512, 512},
			{14, 14, 512, 512}
	};

	sgx_time_t start;
	sgx_time_t end;
	printf("=================Conv2D==================\n");

	for (int i=0; i<num_conv_benchamrks; i++) {
		int batch = 1;
		int h = conv_sizes[i][0];
		int w = conv_sizes[i][1];
		int ch_in = conv_sizes[i][2];
		int ch_out = conv_sizes[i][3];

		X_data = mem_pool->alloc<float>(h * w * ch_in);
		W_data = mem_pool->alloc<float>(3 * 3 * ch_in * ch_out);
		Z_data = mem_pool->alloc<float>(h * w * ch_out);
		r_left_data = mem_pool->alloc<double>(REPS * h * w);
		r_right_data = mem_pool->alloc<double>(REPS * ch_out);
		W_r_data = mem_pool->alloc<float>(REPS * h * w * ch_in);

		conv2d_compute(batch, h, w, ch_in, ch_out, device_ptr);
		start = get_time();
		for(int i=0; i<4; i++) {
			conv2d_compute(batch, h, w, ch_in, ch_out, device_ptr);
		}
		end = get_time();
		printf("conv2d (%d, %d, %d, %d): %.8f\n", batch, h, w, ch_in, get_elapsed_time(start, end)/4);

		conv2d_verif_nopreproc(batch, h, w, ch_in, ch_out, mem_pool, device_ptr);
		start = get_time();
		for(int i=0; i<4; i++) {
			conv2d_verif_nopreproc(batch, h, w, ch_in, ch_out, mem_pool, device_ptr);
		}
		end = get_time();
		printf("conv2d_verif_nopreproc (%d, %d, %d, %d): %.8f\n", batch, h, w, ch_in, get_elapsed_time(start, end)/4);

		conv2d_verif_preproc(batch, h, w, ch_in, ch_out, device_ptr);
		start = get_time();
		for(int i=0; i<4; i++) {
			conv2d_verif_preproc(batch, h, w, ch_in, ch_out, device_ptr);
		}
		end = get_time();
		printf("conv2d_verif_preproc (%d, %d, %d, %d): %.8f\n", batch, h, w, ch_in, get_elapsed_time(start, end)/4);

		mem_pool->release(X_data);
		mem_pool->release(W_data);
		mem_pool->release(Z_data);
		mem_pool->release(r_left_data);
		mem_pool->release(r_right_data);
		mem_pool->release(W_r_data);

		for (batch = 4; batch <= 16; batch += 4) {

			X_data = mem_pool->alloc<float>(batch * h * w * ch_in);
			W_data = mem_pool->alloc<float>(3 * 3 * ch_in * ch_out);
			Z_data = mem_pool->alloc<float>(batch * h * w * ch_out);
			r_left_data = mem_pool->alloc<double>(REPS * batch);
			r_right_data = mem_pool->alloc<double>(REPS * ch_out);
			W_r_data = mem_pool->alloc<float>(REPS * h * w * ch_in);

			conv2d_verif_batch(batch, h, w, ch_in, ch_out, device_ptr);
			start = get_time();
			for(int i=0; i<4; i++) {
				conv2d_verif_batch(batch, h, w, ch_in, ch_out, device_ptr);
			}
			end = get_time();
			printf("conv2d_verif_batched (%d, %d, %d, %d): %.8f\n", batch, h, w, ch_in, get_elapsed_time(start, end)/(4*batch));

			mem_pool->release(X_data);
			mem_pool->release(W_data);
			mem_pool->release(Z_data);
			mem_pool->release(r_left_data);
			mem_pool->release(r_right_data);
			mem_pool->release(W_r_data);
		}

		printf("------------------------------------------------------\n");
	}
}

template <typename T>
void matrix_vector_prod(float* X, T* v, double* res, int n) {
	#pragma omp parallel
	{
		double v2_private[REPS * n] = {};
		int i,j;
		for (i = 0; i < n; i++) {
			#pragma omp for
			for (j = 0; j < n; j++) {
				for (int r = 0; r<REPS; r++) {
					v2_private[r*n + i] += static_cast<double>(X[i*n + j]) * static_cast<double>(v[REPS * n + j]);
				}
			}
		}
		#pragma omp critical
		{
			for(i=0; i<REPS * n; i++) res[i] += v2_private[i];
		}
	}
}

void dense_compute(int batch, int n, int m, void* device_ptr) {

	if (batch == 1) {
		VectorMap<float> X(X_data, n);
		MatrixMap<float> W(W_data, n, m);
		VectorMap<float> Z(Z_data, m);

		Z = X * W;
	} else {
		MatrixMap<float> X(X_data, batch, n);
		MatrixMap<float> W(W_data, n, m);
		MatrixMap<float> Z(Z_data, batch, m);

		Z = X * W;
	}
}

void dense_verif_preproc(int batch, int n, int m, void* device_ptr) {

	if (batch == 1) {
		VectorMap<float> X(X_data, n);
		MatrixMap<float> W_r(W_r_data, n, REPS);
		VectorMap<float> Z(Z_data, m);
		MatrixMap<double> r_right(r_right_data, m, REPS);

		Matrix<double> out1 = X.template cast<double>() * W_r.template cast<double>();
		Matrix<double> out2 = Z.template cast<double>() * r_right;
		out1.resize(0,0);
		out2.resize(0,0);
	} else {
		MatrixMap<float> X(X_data, batch, n);
		MatrixMap<float> W_r(W_r_data, n, REPS);
		MatrixMap<float> Z(Z_data, batch, m);
		MatrixMap<double> r_right(r_right_data, m, REPS);

		#ifdef EIGEN_USE_THREADS
		Tensor<double, 1> out1(REPS * n);
		matrix_vector_prod(X_data, W_r_data, out1.data(), n);
		matrix_vector_prod(Z_data, r_right_data, out1.data(), n);
		out1.resize(0);
		#else
		Matrix<double> out1 = X.template cast<double>() * W_r.template cast<double>();
		Matrix<double> out2 = Z.template cast<double>() * r_right;
		out1.resize(0,0);
		out2.resize(0,0);
		#endif
	}
}

void dense_verif_batch_left(int batch, int n, int m, void* device_ptr) {

	MatrixMap<float> X(X_data, batch, n);
	MatrixMap<float> W(W_data, n, m);
	MatrixMap<float> Z(Z_data, batch, m);
	MatrixMap<double> r_left(r_left_data, REPS, batch);

	Matrix<double> out1 = (r_left * X.template cast<double>()) * W.template cast<double>();
	Matrix<double> out2 = (r_left * Z.template cast<double>());
	out1.resize(0,0);
	out2.resize(0,0);
}

void dense_verif_batch_right(int batch, int n, int m, void* device_ptr) {

	MatrixMap<float> X(X_data, batch, n);
	MatrixMap<float> W(W_data, n, m);
	MatrixMap<float> Z(Z_data, batch, m);
	MatrixMap<double> r_right(r_right_data, m, REPS);

	#ifdef EIGEN_USE_THREADS
	Tensor<double, 1> out1(REPS * n);
	Tensor<double, 1> W_r(REPS * n);
	matrix_vector_prod(W_data, r_right_data, W_r.data(), n);
	matrix_vector_prod(X_data, W_r.data(), out1.data(), n);
	matrix_vector_prod(Z_data, r_right_data, out1.data(), n);
	out1.resize(0);
	W_r.resize(0);
	#else
	Matrix<double> W_r = (W.template cast<double>() * r_right);
	Matrix<double> out1 = X.template cast<double>() * W_r;
	Matrix<double> out2 = Z.template cast<double>() * r_right;
	out1.resize(0,0);
	out2.resize(0,0);
	out1.resize(0,0);
	W_r.resize(0,0);
	#endif

}

void benchmark_dense(MemPool* mem_pool, void* device_ptr) {

	#ifndef USE_SGX
	int num_dense_benchamrks = 6;
	const unsigned int mat_sizes[num_dense_benchamrks] = {
			256, 512, 1024, 2048, 4096, 8192
	};
	#else
	int num_dense_benchamrks = 5;
	const unsigned int mat_sizes[num_dense_benchamrks] = {
			256, 512, 1024, 2048, 4096
	};
	#endif

	sgx_time_t start;
	sgx_time_t end;
	printf("=================Dense==================\n");

	for (int i=0; i<num_dense_benchamrks; i++) {
		int n = mat_sizes[i];

		int batch = n;
		X_data = mem_pool->alloc<float>(batch * n);
		W_data = mem_pool->alloc<float>(n * n);
		Z_data = mem_pool->alloc<float>(batch * n);
		r_right_data = mem_pool->alloc<double>(REPS * n);
		r_left_data = mem_pool->alloc<double>(REPS * batch);
		W_r_data = mem_pool->alloc<float>(REPS * n);

		dense_compute(batch, n, n, device_ptr);
		start = get_time();
		dense_compute(batch, n, n, device_ptr);
		end = get_time();
		printf("dense (%d, %d): %.8f\n", batch, n, get_elapsed_time(start, end)/batch);

		dense_verif_preproc(batch, n, n, device_ptr);
		start = get_time();
		dense_verif_preproc(batch, n, n, device_ptr);
		end = get_time();
		printf("dense_verif_preproc (%d, %d): %.8f\n", batch, n, get_elapsed_time(start, end)/batch);

		dense_verif_batch_left(batch, n, n, device_ptr);
		start = get_time();
		dense_verif_batch_left(batch, n, n, device_ptr);
		end = get_time();
		printf("dense_verif_batched_left (%d, %d): %.8f\n", batch, n, get_elapsed_time(start, end)/batch);

		dense_verif_batch_right(batch, n, n, device_ptr);
		start = get_time();
		dense_verif_batch_right(batch, n, n, device_ptr);
		end = get_time();
		printf("dense_verif_batched_right (%d, %d): %.8f\n", batch, n, get_elapsed_time(start, end)/batch);

		mem_pool->release(X_data);
		mem_pool->release(W_data);
		mem_pool->release(Z_data);
		mem_pool->release(r_right_data);
		mem_pool->release(r_left_data);
		mem_pool->release(W_r_data);

		printf("------------------------------------------------------\n");
	}
}

void sep_conv2d_compute(int batch, int h, int w, int ch_in, int ch_out, void* device_ptr) {
#ifdef EIGEN_USE_THREADS
	Eigen::ThreadPoolDevice* d = static_cast<Eigen::ThreadPoolDevice*>(device_ptr);
#endif

	assert(batch == 1);
	DepthwiseArgs args;
	args.in_rows = h;
	args.in_cols = w;
	args.in_depth = ch_in;
	args.filter_rows = 3;
	args.filter_cols = 3;
	args.depth_multiplier = 1;
	args.stride = 1;
	args.out_rows = h;
	args.out_cols = w;
	args.out_depth = ch_in;

	TensorMap<float, 4> X(X_data, batch, h, w, ch_in);
	TensorMap<float, 4> W(W_data, 1, 1, ch_in, ch_out);
	TensorMap<float, 4> Z(Z_data, batch, h, w, ch_out);

	Eigen::array<std::pair<int, int>, 4> paddings_;
	paddings_[0] = std::make_pair(0, 0);
	paddings_[1] = std::make_pair(1, 1);
	paddings_[2] = std::make_pair(1, 1);
	paddings_[3] = std::make_pair(0, 0);

#ifndef EIGEN_USE_THREADS
	Tensor<float, 4> padded = X.pad(paddings_);
	depthwise_conv<float>(args, padded.data(), W_data, Z_data);
	Z = Eigen::SpatialConvolution(X, W, 1, 1, Eigen::PaddingType::PADDING_SAME);
#else
	Tensor<float, 4> padded(batch, h+2, w+2, ch_in);
	padded.device(*d) = X.pad(paddings_);
	depthwise_conv<float>(args, padded.data(), W_data, Z_data);
	Z.device(*d) = Eigen::SpatialConvolution(X, W, 1, 1, Eigen::PaddingType::PADDING_SAME);
#endif
}

void sep_conv2d_verif_preproc(int batch, int h, int w, int ch_in, int ch_out, void* device_ptr) {
	return conv2d_verif_preproc(batch, h, w, ch_in, ch_out, device_ptr);
}

void sep_conv2d_verif_preproc_intermediate_act(int batch, int h, int w, int ch_in, int ch_out, void* device_ptr) {
	Tensor<double, 1> temp(REPS*ch_in);

	// verif depthwise conv
	for (int i=0; i<h*w; i++) {
		for (int j=0; j<ch_in; j++) {
			for (int r=0; r<REPS; r++) {
				temp[r*ch_in + j] += static_cast<double>(X_data[i*ch_in + j]) *
									 static_cast<double>(W_r_data[r*h*w*ch_in + i*ch_in + j]);
			}
		}
	}

	int out_image_size = h*w;
	__m256 z;
	__m256d z0, z1;
	for (int i=0; i<out_image_size; i++) {
		__m256d rl[REPS];
		for (int r=0; r<REPS; r++) {
			rl[r] = _mm256_broadcast_sd(r_left_data + (r*out_image_size + i));
		}

		for (int j=0; j<ch_out; j+=8) {
			z = _mm256_load_ps(Z_data + (i*ch_out + j));
			extract_two_doubles(z, z0, z1);

			for (int r=0; r<REPS; r++) {
				__m256d prod0 = _mm256_mul_pd(z0, rl[r]);
				__m256d prod1 = _mm256_mul_pd(z1, rl[r]);
				__m256d curr0 = _mm256_load_pd(temp.data() + (r*ch_out + j));
				__m256d curr1 = _mm256_load_pd(temp.data() + (r*ch_out + j + 4));
				_mm256_store_pd(temp.data() + (r*ch_out + j), _mm256_sub_pd(curr0, prod0));
				_mm256_store_pd(temp.data() + (r*ch_out + j + 4), _mm256_sub_pd(curr1, prod1));
			}
		}
	}

	// verif pointwise conv
	conv2d_verif_preproc(batch, h, w, ch_in, ch_out, device_ptr);
}

void sep_conv2d_verif_batch(int batch, int h, int w, int ch_in, int ch_out, void* device_ptr) {
	return conv2d_verif_batch(batch, h, w, ch_in, ch_out, device_ptr);
}

void sep_conv2d_verif_batch_intermediate_act(int batch, int h, int w, int ch_in, int ch_out, void* device_ptr) {
	Tensor<double, 1> temp(REPS*h*w*ch_out);
	Tensor<double, 1> out1(REPS*h*w*ch_out);

	temp.setZero();

	for (int i=0; i<batch; i++) {
		for(int j=0; j<h*w*ch_in; j++) {
			for(int r=0; r<REPS; r++) {
				temp[r*h*w*ch_in + j] += static_cast<double>(X_data[i*h*w*ch_in + j]) * r_right_data[r*batch + i];
			}
		}
	}

	DepthwiseArgs args;
	args.in_rows = h;
	args.in_cols = w;
	args.in_depth = ch_in;
	args.filter_rows = 3;
	args.filter_cols = 3;
	args.depth_multiplier = 1;
	args.stride = 1;
	args.pad_rows = 1;
	args.pad_cols = 1;
	args.out_rows = h;
	args.out_cols = w;
	args.out_depth = ch_in;
	args.batch = REPS;
	TensorMap<float, 1> kernel_(W_data, args.filter_rows * args.filter_cols * args.in_depth);
	Tensor<double, 1> kernel_dbl = kernel_.template cast<double>();
	depthwise_conv<double>(args, temp.data(), kernel_dbl.data(), out1.data());

	for (int i=0; i<batch; i++) {
		for(int j=0; j<h*w*ch_out; j++) {
			for(int r=0; r<REPS; r++) {
				out1[r*h*w*ch_out + j] -= static_cast<double>(Z_data[i*h*w*ch_out + j]) * r_right_data[r*batch + i];
			}
		}
	}

	// pointwise convolution
	conv2d_verif_batch(batch, h, w, ch_in, ch_out, device_ptr, 1, 1);
}

void benchmark_separable(MemPool* mem_pool, void* device_ptr) {

	int num_conv_benchamrks = 4;
	const unsigned int conv_sizes[num_conv_benchamrks][4] = {
			{112, 112, 32, 32},
			{56, 56, 128, 128},
			{28, 28, 256, 256},
			{14, 14, 512, 512}
	};

	sgx_time_t start;
	sgx_time_t end;
	printf("=================Conv2D Sep==================\n");

	for (int i=0; i<num_conv_benchamrks; i++) {
		int batch = 1;
		int h = conv_sizes[i][0];
		int w = conv_sizes[i][1];
		int ch_in = conv_sizes[i][2];
		int ch_out = conv_sizes[i][3];

		X_data = mem_pool->alloc<float>(h * w * ch_in);
		W_data = mem_pool->alloc<float>(3 * 3 * ch_in * ch_out);
		Z_data = mem_pool->alloc<float>(h * w * ch_out);
		r_left_data = mem_pool->alloc<double>(REPS * h * w);
		r_right_data = mem_pool->alloc<double>(REPS * ch_out);
		W_r_data = mem_pool->alloc<float>(REPS * h * w * ch_in);


		sep_conv2d_compute(batch, h, w, ch_in, ch_out, device_ptr);
		start = get_time();
		for(int i=0; i<4; i++) {
			sep_conv2d_compute(batch, h, w, ch_in, ch_out, device_ptr);
		}
		end = get_time();
		printf("sep_conv2d (%d, %d, %d, %d): %.8f\n", batch, h, w, ch_in, get_elapsed_time(start, end)/4);

		sep_conv2d_verif_preproc(batch, h, w, ch_in, ch_out, device_ptr);
		start = get_time();
		for(int i=0; i<4; i++) {
			sep_conv2d_verif_preproc(batch, h, w, ch_in, ch_out, device_ptr);
		}
		end = get_time();
		printf("sep_conv2d_verif_preproc (%d, %d, %d, %d): %.8f\n", batch, h, w, ch_in, get_elapsed_time(start, end)/4);

		sep_conv2d_verif_preproc_intermediate_act(batch, h, w, ch_in, ch_out, device_ptr);
		start = get_time();
		for(int i=0; i<4; i++) {
			sep_conv2d_verif_preproc_intermediate_act(batch, h, w, ch_in, ch_out, device_ptr);
		}
		end = get_time();
		printf("sep_conv2d_verif_preproc_inter (%d, %d, %d, %d): %.8f\n", batch, h, w, ch_in, get_elapsed_time(start, end)/4);

		mem_pool->release(X_data);
		mem_pool->release(W_data);
		mem_pool->release(Z_data);
		mem_pool->release(r_left_data);
		mem_pool->release(r_right_data);
		mem_pool->release(W_r_data);

		for (batch = 4; batch <= 32; batch *= 2) {

			X_data = mem_pool->alloc<float>(batch * h * w * ch_in);
			W_data = mem_pool->alloc<float>(3 * 3 * ch_in * ch_out);
			Z_data = mem_pool->alloc<float>(batch * h * w * ch_out);
			r_left_data = mem_pool->alloc<double>(REPS * batch);
			r_right_data = mem_pool->alloc<double>(REPS * ch_out);
			W_r_data = mem_pool->alloc<float>(REPS * h * w * ch_in);

			sep_conv2d_verif_batch(batch, h, w, ch_in, ch_out, device_ptr);
			start = get_time();
			for(int i=0; i<4; i++) {
				sep_conv2d_verif_batch(batch, h, w, ch_in, ch_out, device_ptr);
			}
			end = get_time();
			printf("sep_conv2d_verif_batched (%d, %d, %d, %d): %.8f\n", batch, h, w, ch_in, get_elapsed_time(start, end)/(4*batch));

			sep_conv2d_verif_batch_intermediate_act(batch, h, w, ch_in, ch_out, device_ptr);
			start = get_time();
			for(int i=0; i<4; i++) {
				sep_conv2d_verif_batch_intermediate_act(batch, h, w, ch_in, ch_out, device_ptr);
			}
			end = get_time();
			printf("sep_conv2d_verif_batched_inter (%d, %d, %d, %d): %.8f\n", batch, h, w, ch_in, get_elapsed_time(start, end)/(4*batch));

			mem_pool->release(X_data);
			mem_pool->release(W_data);
			mem_pool->release(Z_data);
			mem_pool->release(r_left_data);
			mem_pool->release(r_right_data);
			mem_pool->release(W_r_data);
		}

		printf("------------------------------------------------------\n");
	}
}


void benchmark(int n_threads) {
	MemPool* mem_pool = new MemPool(0, 0);
	TIMING = true;

	printf("in benchmark with %d threads\n", n_threads);

#ifdef EIGEN_USE_THREADS
	Eigen::ThreadPool pool(n_threads);
	Eigen::ThreadPoolDevice device(&pool, n_threads);
	Eigen::setNbThreads(n_threads);
	void* device_ptr = (void*) &device;
	omp_set_num_threads(n_threads);
#else
	assert(n_threads == 1);

	#ifndef USE_SGX
	omp_set_num_threads(1);
	#endif

	void* device_ptr = nullptr;
#endif

	benchmark_dense(mem_pool, device_ptr);
	benchmark_conv(mem_pool, device_ptr);
	benchmark_separable(mem_pool, device_ptr);
}

#endif
