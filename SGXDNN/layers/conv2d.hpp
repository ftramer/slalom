#ifndef SGXDNN_CONV2D_H_
#define SGXDNN_CONV2D_H_

#define EIGEN_USE_TENSOR

#include <stdio.h>
#include <iostream>
#include <string>
#include <type_traits>
#include <assert.h>

#include "../mempool.hpp"
#include "../utils.hpp"
#include "../Crypto.h"
#include "layer.hpp"
#include "activation.hpp"
#include "eigen_spatial_convolutions.h"
#include <cmath>
#include "immintrin.h"

#ifndef USE_SGX
#include <chrono>
#else
#include "Enclave.h"
#include "sgx_tcrypto.h"
#include "Crypto.h"
#endif

using namespace tensorflow;

namespace SGXDNN
{

	template <class T1, class T2, class T3>
	void conv2d_im2col(const T1* input_data, int input_batches, int input_height, int input_width, int input_depth, 
					   T2* filter_data, int filter_height, int filter_width, int filter_count, 
					   int stride_rows, int stride_cols, Eigen::PaddingType padding, 
					   T3* output_data, int output_height, int output_width, MemPool* mem_pool_);

	template <typename T>
	class Conv2D : public Layer<T>
	{
	public:
		Conv2D(const std::string& name,
			   const array4d input_shape,
               const array4d& kernel_shape,
               const int row_stride,
               const int col_stride,
               const Eigen::PaddingType& padding,
               T* r_left, T* r_right, T* kernel, T* bias,
			   MemPool* mem_pool,
			   bool is_verif_mode,
			   bool verif_preproc,
			   const std::string& activation_type
			   ): Layer<T>(name, input_shape),
			   kernel_shape_(kernel_shape),
               row_stride_(row_stride),
               col_stride_(col_stride),
               padding_(padding),
               r_left_data_(nullptr),
               r_right_data_(nullptr),
               kernel_data_(nullptr),
               bias_data_(nullptr),
               kernel_(NULL, kernel_shape),
               bias_(NULL, kernel_shape[3]),
			   r_left_(NULL, REPS, input_shape[1] * input_shape[2]),
			   r_right_(NULL, kernel_shape[3], REPS),
			   kernel_r_(NULL, kernel_shape[0] * kernel_shape[1] * kernel_shape[2], REPS),
			   bias_r_(NULL, REPS),
			   mem_pool_(mem_pool),
			   verif_preproc_(verif_preproc),
			   activation_type_(activation_type),
			   h(input_shape[1]),
			   w(input_shape[2]),
			   ch_in(kernel_shape[2]),
			   h_out(0),
			   w_out(0),
			   ch_out(kernel_shape[3]),
			   patch_size(kernel_shape[0] * kernel_shape[1]),
			   image_size(input_shape[1] * input_shape[2]),
			   out_image_size(0)
		{
			const int filter_rows = kernel_shape[0];
			const int filter_cols = kernel_shape[1];

			GetWindowedOutputSize(h, filter_rows, row_stride_,
								  padding_, &h_out, &pad_rows_);
			GetWindowedOutputSize(w, filter_cols, col_stride_,
								  padding_, &w_out, &pad_cols_);

			printf("in Conv2D with out_shape = (%d, %d, %d)\n", h_out, w_out, ch_out);
			output_shape_ = {0, h_out, w_out, ch_out};
			output_size_ = h_out * w_out * ch_out;
			input_shape_ = {0, h, w, ch_in};
			input_size_ = h * w * ch_in;
			out_image_size = h_out * w_out;

			if (is_verif_mode)
			{
				// pointers to original weight data
				kernel_data_ = kernel;
				bias_data_ = bias;

				if (!verif_preproc) {
					// copy kernel and bias into enclave
					long kernel_size = kernel_shape[0] * kernel_shape[1] * kernel_shape[2] * kernel_shape[3];
					kernel_data_ = mem_pool_->alloc<T>(kernel_size);
					std::copy(kernel, kernel + kernel_size, kernel_data_);
					new (&kernel_) typename TTypes<T, 4>::Tensor(kernel_data_, kernel_shape);
		
					long bias_size = kernel_shape[3];
					bias_data_ = mem_pool_->alloc<T>(bias_size);
					std::copy(bias, bias + bias_size, bias_data_);
					new (&bias_) typename TTypes<T>::ConstVec(bias_data_, kernel_shape[3]);
				} else {

					// load pre-computed r_left, r_right, W_r and b_r
					// TODO actually compute these things in the enclave
					if (filter_rows == 1 and filter_cols == 1) {

						// pointwise convolution
						r_right_data_ = mem_pool_->alloc<double>(ch_out * REPS);
                        kernel_r_data_ = mem_pool_->alloc<T>(ch_in * REPS);
                        bias_r_data_ = mem_pool_->alloc<double>(REPS);

						// Tensor maps
                        new (&r_right_) TensorMap<double, 2>(r_right_data_, REPS, ch_out);
                        new (&kernel_r_) TensorMap<T, 2>(kernel_r_data_, REPS, ch_in);
                        new (&bias_r_) TensorMap<double, 1>(bias_r_data_, REPS);

                        std::copy(r_right, r_right + REPS * ch_out, r_right_data_);
                        std::copy(kernel, kernel + REPS * ch_in, kernel_r_data_);
                        std::copy(bias, bias + REPS, bias_r_data_);

						Tensor<double, 0> sum;
						sum = r_right_.sum();
						printf("r_right: %f\n", sum.data()[0]);
						sum = kernel_r_.template cast<double>().sum();
						printf("kernel_r: %f\n", sum.data()[0]);
						sum = bias_r_.sum();
						printf("bias_r: %f\n", sum.data()[0]);

					} else {
						// standard convolution

						r_left_data_ = mem_pool_->alloc<double>(REPS * out_image_size);
						r_right_data_ = mem_pool_->alloc<double>(ch_out * REPS);
						kernel_r_data_ = mem_pool_->alloc<T>(image_size * ch_in * REPS);
						bias_r_data_ = mem_pool_->alloc<double>(REPS);

						// Tensor maps
						new (&r_left_) TensorMap<double, 2>(r_left_data_, REPS, out_image_size);
						new (&r_right_) TensorMap<double, 2>(r_right_data_, REPS, ch_out);
						new (&kernel_r_) TensorMap<T, 2>(kernel_r_data_, REPS, image_size * ch_in);
						new (&bias_r_) TensorMap<double, 1>(bias_r_data_, REPS);

						std::copy(r_left, r_left + REPS * out_image_size, r_left_data_);
						std::copy(r_right, r_right + REPS * ch_out, r_right_data_);
						std::copy(kernel, kernel + REPS * image_size * ch_in, kernel_r_data_);
						std::copy(bias, bias + REPS, bias_r_data_);

						Tensor<double, 0> sum;
						sum = r_left_.sum();
						printf("r_left: %f\n", sum.data()[0]);
						sum = r_right_.sum();
						printf("r_right: %f\n", sum.data()[0]);
						sum = kernel_r_.template cast<double>().sum();
						printf("kernel_r: %f\n", sum.data()[0]);
						sum = bias_r_.sum();
						printf("bias_r: %f\n", sum.data()[0]);
					}
				}
			} 
			else
			{
				long kernel_size = kernel_shape[0] * kernel_shape[1] * kernel_shape[2] * kernel_shape[3];

				lazy_load_ = false;
				#ifdef USE_SGX
				if (mem_pool_->allocated_bytes >= 50 * 1024 * 1024) {
					lazy_load_ = true;
					printf("lazy loading convolution of size %ld\n", kernel_size);
				}
				#endif
				
				// copy kernel and bias
				if (lazy_load_) {
					kernel_data_ = kernel;
					mac = new MAC();
				} else {
					kernel_data_ = mem_pool_->alloc<T>(kernel_size); 
					std::copy(kernel, kernel + kernel_size, kernel_data_);
					new (&kernel_) typename TTypes<T, 4>::Tensor(kernel_data_, kernel_shape);
				}

				long bias_size = kernel_shape[3];
				bias_data_ = new T[bias_size];
				std::copy(bias, bias + bias_size, bias_data_);
				new (&bias_) typename TTypes<T>::ConstVec(bias_data_, kernel_shape[3]);
			}
		}

		array4d output_shape() override
		{
			return output_shape_;
		}

		int output_size() override
		{
			return output_size_;
		}

		int num_linear() override
		{
			return 1;
		}

		void set_activation_type(const std::string act) {
			activation_type_ = act;
		}

		// res_x = b_r
		inline void preproc_verif_pointwise_bias() {
			for (int i=0; i<h_out*w_out; i++) {
				for(int r=0; r<REPS; r++) {
					res_x[i * REPS + r] = bias_r_.data()[r];
				}
			}
		}

		// inner loop of X*W_r
		inline void preproc_verif_pointwise_X_inner(__m256 x, int i, int j) {
			__m256d x0, x1, kr0, kr1;
			extract_two_doubles(x, x0, x1);

			for(int r=0; r<REPS; r++) {
				load_two_doubles(kernel_r_data_ + (r*ch_in + j), kr0, kr1);
				res_x[i * REPS + r] += double_dot_prod(x0, x1, kr0, kr1);
			}

		}

		// Compute res_x = (b*r) + X * (W*r) where b_r=(b*r) and W_r=(W*r) are precomputed
		inline void preproc_verif_pointwise_X(T* input) {
			assert(ch_in % 8 == 0);
			preproc_verif_pointwise_bias();

			// X*W_r + b_r => shape is (h*w, REPS)
			for (int i=0; i<h_out; i++) {
			    for (int j=0; j<w_out; j++) {
				    for(int c=0; c<ch_in; c+=8) {
				        int offset_i = i * row_stride_;
				        int offset_j = j * col_stride_;
				        int offset = (offset_i*w*ch_in + offset_j*ch_in + c);
					    preproc_verif_pointwise_X_inner(_mm256_load_ps(input + offset), i*w_out+j, c);
					}
				}
			}
		}

		// inner loop of Z * r_right
		inline void preproc_verif_pointwise_Z_inner(__m256 z, int i, int j) {
			__m256d z0, z1, rr0, rr1;
			extract_two_doubles(z, z0, z1);

			for (int r=0; r<REPS; r++) {
				load_two_doubles(r_right_data_ + (r*ch_out + j), rr0, rr1);
				res_z[i*REPS + r] += double_dot_prod(z0, z1, rr0, rr1);
			}
		}

		// compute Z * r_right and activation(Z) in a single loop
		template <typename F>
		inline void preproc_verif_pointwise_Z_memfused(F func, T* extra_data, T* output) {
			assert(ch_out % 8 == 0);
			__m256 z, relu;

			for (int i=0; i<out_image_size; i++) {
				for (int j=0; j<ch_out; j+=8) {
					z = _mm256_load_ps(extra_data + (i*ch_out + j));

					relu = func(z);
					_mm256_store_ps(output + (i*ch_out + j), relu);

					preproc_verif_pointwise_Z_inner(z, i, j);
				}
			}
		}

		// compute Z * r_right
		template <typename F>
		inline void preproc_verif_pointwise_Z(F func, T* extra_data) {
			assert(ch_out % 8 == 0);
			__m256 z, relu;

			for (int i=0; i<out_image_size; i++) {
				for (int j=0; j<ch_out; j+=8) {
					z = _mm256_load_ps(extra_data + (i*ch_out + j));
					relu = func(z);
					preproc_verif_pointwise_Z_inner(z, i, j);
				}
			}
		}

		// res_x = b_r
		inline void preproc_verif_bias() {
			for (int r=0; r<REPS; r++) {
				res_x[r] = bias_r_.data()[r];
			}
		}

		// inner loop of X*W_r
		inline void preproc_verif_X_inner(__m256 x, int i, int j, __m256d* temp) {
			__m256d x0, x1, kr0, kr1;
			extract_two_doubles(x, x0, x1);

			for (int r=0; r<REPS; r++) {
				load_two_doubles(kernel_r_data_ + (r * image_size * ch_in + i * ch_in + j), kr0, kr1);
				//res_x[r] += double_dot_prod(x0, x1, kr0, kr1);
				temp[r] = double_dot_prod_fmadd(x0, x1, kr0, kr1, temp[r]);
			}
		}

		// Compute res_x = (r_left*b*r_right) + r_left * X * (W*r_right) where b_r and W_r are precomputed
		inline void preproc_verif_X(T* input) {
			preproc_verif_bias();
			__m256d x0, x1, kr0, kr1;

			if (ch_in % 8 != 0) {
				assert((image_size * ch_in) % 8 == 0);

				for (int k=0; k<image_size * ch_in; k+=8) {
					load_two_doubles(input + k, x0, x1);

					for (int r=0; r<REPS; r++) {
						load_two_doubles(kernel_r_data_ + (r * image_size * ch_in + k), kr0, kr1);
						res_x[r] += double_dot_prod(x0, x1, kr0, kr1);
					}
				}
			} else {

				__m256d temp[REPS];
				for (int r=0; r<REPS; r++) {
					temp[r] = _mm256_setzero_pd();	
				}

				for (int i=0; i<image_size; i++) {
					for (int j=0; j<ch_in; j += 8) {
						preproc_verif_X_inner(_mm256_load_ps(input + i * ch_in + j), i, j, temp);
					}
				}
				
				for (int r=0; r<REPS; r++) {
					res_x[r] += sum_m256d(temp[r]);
				}
			}
		}

		// inner loop of r_left * (Z * r_right)
		inline void preproc_verif_Z_inner( __m256 z, int i, int j, __m256d* temp) {
			__m256d z0, z1, rr0, rr1;
			extract_two_doubles(z, z0, z1);

			for (int r=0; r<REPS; r++) {
				load_two_doubles(r_right_data_ + (r*ch_out + j), rr0, rr1);
				temp[r] = double_dot_prod_fmadd(z0, z1, rr0, rr1, temp[r]);
			}
		}

		// outer loop of r_left * (Z * r_right)
		inline void preproc_verif_Z_outer(int i, __m256d* temp) {
			for (int r=0; r<REPS; r++) {
				double rl = r_left_.data()[r*out_image_size + i];
				//double t = res_z_temp[r];
				double t = sum_m256d(temp[r]);
				REDUCE_MOD(t);
				res_z[r] += rl * t;
			}
		}

		// compute r_left * Z * r_right and activation(Z) in a single loop
		template <typename F>
		inline void preproc_verif_Z_memfused(F func, T* extra_data, T* output) {
			assert(ch_out % 8 == 0);
			__m256 z, relu;

			__m256d temp[REPS];
			for (int i=0; i<out_image_size; i++) {
				for (int r=0; r<REPS; r++) {
    	        	temp[r] = _mm256_setzero_pd();
        	    }

				for (int j=0; j<ch_out; j+=8) {
					z = _mm256_load_ps(extra_data + (i*ch_out + j));

					relu = func(z);
					_mm256_store_ps(output + (i*ch_out + j), relu);

					preproc_verif_Z_inner(z, i, j, temp);
				}

				preproc_verif_Z_outer(i, temp);
			}
		}

		// compute r_left * Z * r_right
		template <typename F>
		inline void preproc_verif_Z(F func, T* extra_data) {
			assert(ch_out % 8 == 0);
			__m256 z;

			__m256d temp[REPS];
			for (int i=0; i<out_image_size; i++) {
				for (int r=0; r<REPS; r++) {
                    temp[r] = _mm256_setzero_pd();
                }
				for (int j=0; j<ch_out; j+=8) {
					z = _mm256_load_ps(extra_data + (i*ch_out + j));
					preproc_verif_Z_inner(z, i, j, temp);
				}

				preproc_verif_Z_outer(i, temp);
			}
		}

		int h;
		int w;
		int ch_in;
		int h_out;
		int w_out;
		int ch_out;
		int patch_size;
		int image_size;
		int out_image_size;

		double* res_x;
		double* res_z;
		double* res_z_temp;

		// verification data
		double* r_left_data_;
		double* r_right_data_;
		T* kernel_r_data_;		// keep this in single precision for SGX
		double* bias_r_data_;

	protected:

		TensorMap<T, 4> apply_impl(TensorMap<T, 4> input, void* device_ptr = NULL, bool release_input=true) override
		{
			#ifdef EIGEN_USE_THREADS
			Eigen::ThreadPoolDevice* d = static_cast<Eigen::ThreadPoolDevice*>(device_ptr);
			#endif

			T* kernel_temp;
			if (lazy_load_) {
				long kernel_size = kernel_shape_[0] * kernel_shape_[1] * kernel_shape_[2] * kernel_shape_[3];
                kernel_temp = mem_pool_->alloc<T>(kernel_size); 
                std::copy(kernel_data_, kernel_data_ + kernel_size, kernel_temp);
                new (&kernel_) typename TTypes<T, 4>::Tensor(kernel_temp, kernel_shape_);
				// TODO actually check mac...
                Tag tag = mac->mac((uint8_t*) kernel_temp, kernel_size * sizeof(T));
			}
  
			const int batch = input.dimension(0);
			output_shape_[0] = batch;

			// allocate memory to store the output
			T* output_mem_ = mem_pool_->alloc<T>(batch * output_size_);
			auto output_map = TensorMap<T, 4>(output_mem_, output_shape_);
		
			sgx_time_t start = get_time();
			output_map = Eigen::SpatialConvolution(input, kernel_, col_stride_, row_stride_, padding_);
			sgx_time_t end = get_time();

			if (TIMING) { printf("convd (%ld x %ld x %ld) took %.4f seconds\n", input.dimension(1), input.dimension(2), input.dimension(3), get_elapsed_time(start, end)); };

			if (lazy_load_) {
				mem_pool_->release(kernel_temp);
			}

			// add bias
			const int bias_size = bias_.dimension(0);
			const int rest_size = output_map.size() / bias_size;
			Eigen::DSizes<int, 1> one_d(output_map.size());
			Eigen::DSizes<int, 1> bcast(rest_size);
	
			output_map.reshape(one_d) = output_map.reshape(one_d) + bias_.broadcast(bcast).reshape(one_d);
			if (release_input) {
				mem_pool_->release(input.data());
			}

			return output_map;
		}

		TensorMap<T, 4> fwd_verify_impl(TensorMap<T, 4> input, float** aux_data, int linear_idx, void* device_ptr = NULL, bool release_input = true) override
        {
			#ifdef EIGEN_USE_THREADS
            Eigen::ThreadPoolDevice* d = static_cast<Eigen::ThreadPoolDevice*>(device_ptr);
			#endif
			
			float* extra_data = aux_data[linear_idx];

	    	const long batch = input.dimension(0);
            output_shape_[0] = batch;
            input_shape_[0] = batch;

			if (!verif_preproc_)
			{
				// batched verification
				if (batch > 1) 
				{
					bool simple_comp = false;
					#ifdef USE_SGX
					simple_comp = (32 <= ch_in) && (ch_in <= 128);
					#endif

					// if the number of input channels is small, we get no savings on SGX so just compute
					if (simple_comp) {
						return apply_impl(input, device_ptr);
					}

					sgx_time_t start = get_time();
					// TODO randomize these
					Tensor<double, 2> r_b(REPS, batch);
					Tensor<double, 2> r_ch(REPS, ch_out);
					r_b.setConstant(1 << 20);
					r_ch.setConstant(1 << 20);

					int max_size = std::max(h*w*ch_in, h_out*w_out*ch_out);
					Tensor<double, 1> temp(REPS * max_size); 
					temp.setZero();

					// compute (r_b * X) mod p
					for (int i=0; i<batch; i++) {
						for(int j=0; j<h*w*ch_in; j++) {
							for(int r=0; r<REPS; r++) {
								temp.data()[r*h*w*ch_in + j] += static_cast<double>(input.data()[i*h*w*ch_in + j]) *
															    r_b.data()[r*batch + i];
							}
						}
					}

					// reduce mod p
					for (int r=0; r<REPS; r++) {
						for (int j=0; j<h*w*ch_in; j++) {
							REDUCE_MOD(temp.data()[r*h*w*ch_in + j]);
						}
					}
		
					// no need for the input anymore
					if (release_input) {
						mem_pool_->release(input.data());
					}

					sgx_time_t loop1 = get_time();
					if (TIMING) {
						printf("after loop1 in %.4f s\n", get_elapsed_time(start, loop1));
					}

					// compute W*r_ch and sum(r_b) * b * r_ch
					array2d W_dims2d{{kernel_.dimension(0) * kernel_.dimension(1) * kernel_.dimension(2), kernel_.dimension(3)}};
					array4d W_dims4d{{kernel_.dimension(0), kernel_.dimension(1), kernel_.dimension(2), 1}};

					// Compute Conv(r_b * X, W*r_ch)
					Tensor<double, 1> out1(REPS * h_out * w_out);
					for (int i=0; i<REPS; i++) {
						TensorMap<double, 2> r_ch_temp(r_ch.data() + i * ch_out, ch_out, 1); 	
						Tensor<double, 4> W_r_temp = kernel_.reshape(W_dims2d).template cast<double>().contract(r_ch_temp, MATRIX_PRODUCT).reshape(W_dims4d);
						TensorMap<double, 4> X_sum_temp(temp.data() + i*h*w*ch_in, 1, h, w, ch_in);
						TensorMap<double, 4> out_temp(out1.data() + i*h_out*w_out, 1, h_out, w_out, 1);
						out_temp = Eigen::SpatialConvolution(X_sum_temp, W_r_temp, row_stride_, col_stride_, padding_); 
					}

					sgx_time_t conv = get_time();
					if (TIMING) {
						printf("after conv in %.4f s\n", get_elapsed_time(start, conv));
					}
					// copy presumed output into enclave
					T* output_mem_ = mem_pool_->alloc<T>(batch * output_size_);
					#ifndef USE_SGX
					memcpy(output_mem_, extra_data, batch * output_size_ * sizeof(T));
					#else
					std::copy(extra_data, extra_data + batch * output_size_, output_mem_);
					#endif
					auto output_map = TensorMap<T, 4>(output_mem_, output_shape_);

					sgx_time_t alloc = get_time();
					if (TIMING) {
						printf("after alloc in %.4f s\n", get_elapsed_time(start, alloc));
					}
					temp.setZero();
				
					// compute r_b * Z
					for (int i=0; i<batch; i++) {
						for(int j=0; j<h_out*w_out*ch_out; j++) {
							for(int r=0; r<REPS; r++) {
								temp.data()[r*h_out*w_out*ch_out + j] += static_cast<double>(output_map.data()[i*h_out*w_out*ch_out+ j]) *
																		 r_b.data()[r*batch + i];
							}
						}
					}

					sgx_time_t loop2 = get_time();
					if (TIMING) {
						printf("after loop2 in %.4f s\n", get_elapsed_time(start, loop2));
					}
					// temp is of shape (REPS, h_out, w_out, ch_out)	
					// we have r_b * Z, and we want r_b*Z*r_ch - r_b*bias*r_ch
					Tensor<double, 1> bias_r(REPS);
					array1d reps_dim = {1};
					Tensor<double, 1> r_b_sum = r_b.sum(reps_dim);
					for (int r=0; r<REPS; r++) {
						double b_r = 0.0;
						for (int k=0; k<ch_out; k++) {
							double tmp = static_cast<double>(bias_.data()[k]) * r_ch.data()[r*ch_out + k];
							REDUCE_MOD(tmp);
							b_r += tmp;
						}
						bias_r.data()[r] = b_r * r_b_sum.data()[r];
					}

					for (int r=0; r<REPS; r++) {
						for (int j=0; j<h_out*w_out; j++) {
							for (int k=0; k<ch_out; k++) {
								double Z = temp.data()[r*h_out*w_out*ch_out + j*ch_out + k];
								REDUCE_MOD(Z);
								out1.data()[r*h_out*w_out + j] -= Z * r_ch.data()[r*ch_out + k];
							}
							out1.data()[r*h_out*w_out + j] += bias_r.data()[r];
						}
					}

					sgx_time_t rZr = get_time();
					if (TIMING) {
						printf("after r_Z_r in %.4f s\n", get_elapsed_time(start, rZr));
					}
					out1 = out1.unaryExpr([&](const double x) { return mod(x, p_verif); });
					Tensor<bool, 0> eq = (out1 == static_cast<double>(0)).all();
					if (TIMING) {
						printf("eq: %d\n", eq.data()[0]);
					}
					r_b.resize(0,0);
					r_ch.resize(0,0);
					temp.resize(0);
					out1.resize(0);
					sgx_time_t end = get_time();
					double elapsed = get_elapsed_time(start, end);
					if (TIMING) {
						printf("batched convd (%ld x %ld x %ld) took %.4f seconds\n", input.dimension(1), input.dimension(2), input.dimension(3), elapsed);
					}

					return output_map;

				} else if (kernel_.dimension(0) == 1 && kernel_.dimension(0) == 1) {
					// pointwise convolution (i.e., matmul) with single batch

					sgx_time_t start = get_time();

					// TODO randomize
					Tensor<double, 2> r_left(REPS, h*w);
					r_left.setConstant(1 << 20);

					Tensor<double, 2> temp(REPS, ch_in);
					temp.setZero();

					// r*X => shape is (REPS, ch_in)
					for (int i=0; i<h*w; i++) {
						for(int j=0; j<ch_in; j++) {
							for(int r=0; r<REPS; r++) {
								temp.data()[r*ch_in + j] += static_cast<double>(input.data()[i*ch_in + j]) *
															r_left.data()[r*h*w + i];
							}
						}
					}

					if (release_input) {
						mem_pool_->release(input.data());
					}

					// compute (r*X) * W
					array2d W_dims2d{{kernel_.dimension(2), kernel_.dimension(3)}};
					Tensor<double, 2> out1 = temp.contract(kernel_.reshape(W_dims2d).template cast<double>(), MATRIX_PRODUCT);

					temp.resize(0,0);

					// copy the presumably correct product into the enclave
					T* output_mem_ = mem_pool_->alloc<T>(batch * output_size_);
					auto output_map = TensorMap<T, 4>(output_mem_, output_shape_);

					#ifndef USE_SGX
					memcpy(output_mem_, extra_data, batch * output_size_ * sizeof(T));
					#else
					std::copy(extra_data, extra_data + batch * output_size_, output_mem_);
					#endif

					// r*Z => shape is (REPS, ch_out)
					for (int i=0; i<h*w; i++) {
						for(int j=0; j<ch_out; j++) {
							for(int r=0; r<REPS; r++) {
								out1.data()[r*ch_out + j] -= static_cast<double>(output_map.data()[i*ch_out + j]) *
															 r_left.data()[r*h*w + i];
							}
						}
					}

					array1d reps_dim = {1};
					Tensor<double, 1> r_left_sum = r_left.sum(reps_dim);

					// add r*bias
					for(int r=0; r<REPS; r++) {
						for(int j=0; j<ch_out; j++) {
							out1.data()[r*ch_out + j] += r_left_sum.data()[r] * bias_.data()[j];
						}
					}

					// check equality
					Tensor<bool, 0> eq = (out1 == static_cast<double>(0)).all();
					if (TIMING) {
						printf("eq: %d\n", eq.data()[0]);
					}
					sgx_time_t end = get_time();
					double elapsed = get_elapsed_time(start, end);
					if (TIMING) {
						printf("pointwise convd left (%ld x %ld x %ld) took %.4f seconds\n", input.dimension(1), input.dimension(2), input.dimension(3), elapsed);
					}

					out1.resize(0,0);
					r_left.resize(0,0);
					return output_map;

				} else {
					// standard (non-pointwise) convolution

					// define some necessary intermediate shapes
					array2d patch_dims2d{{batch * h * w, patch_size * ch_in}};
					array2d Z_dims2d{{batch * output_shape_[1] * output_shape_[2], output_shape_[3]}};
					array2d Z_dims_out{{batch * output_shape_[1] * output_shape_[2], REPS}};
					array2d W_dims2d{{kernel_.dimension(0) * kernel_.dimension(1) * kernel_.dimension(2), kernel_.dimension(3)}};
					array4d W_dims_out{{kernel_.dimension(0), kernel_.dimension(1), kernel_.dimension(2), REPS}};

					sgx_time_t start = get_time();

					// sample a new random vector (TODO randomize)
					Tensor<double, 2> r(ch_out, REPS);
					r.setConstant(1 << 20);

					// compute W*r and b*r
					Tensor<double, 2> W_r = kernel_.reshape(W_dims2d).template cast<double>().contract(r, MATRIX_PRODUCT);
					Tensor<double, 1> b_r = bias_.template cast<double>().contract(r, INNER_PRODUCT);

					// allocate a temp buffer to store Conv(X, W*r)
					Tensor<double, 1> out1(batch * out_image_size * REPS);
					out1.setZero();

					// compute Conv(X, W*r). Use an im2col implementation tailored for low memory devices
					// as we compute in double precision here
					conv2d_im2col(input.data(), batch, h, w, ch_in,
					   W_r.data(), kernel_.dimension(0), kernel_.dimension(1), REPS,
					   row_stride_, col_stride_, padding_, 
					   out1.data(), output_shape_[1], output_shape_[2], mem_pool_);

					// we don't need the input anymore
					if (release_input) {
						mem_pool_->release(input.data());
					}
					W_r.resize(0, 0);

					// copy the presumably correct product into the enclave
					T* output_mem_ = mem_pool_->alloc<T>(batch * output_size_);
					auto output_map = TensorMap<T, 4>(output_mem_, output_shape_);
					#ifndef USE_SGX
					memcpy(output_mem_, extra_data, batch * output_size_ * sizeof(T));
					#else
					std::copy(extra_data, extra_data + batch * output_size_, output_mem_);
					#endif

					// compute Z*r - b*r
					auto out2 = output_map.reshape(Z_dims2d).template cast<double>().contract(r, MATRIX_PRODUCT);
					const int bias_size = REPS;
					const int rest_size = batch * out_image_size;
					Eigen::DSizes<int, 1> one_d(rest_size * bias_size);
					Eigen::DSizes<int, 1> bcast(rest_size);
					auto out3 = out2.reshape(one_d) - b_r.broadcast(bcast).reshape(one_d);

					// check equality
					Eigen::Tensor<bool, 0, Eigen::RowMajor, Eigen::DenseIndex> eq;
					eq = (out1 == out3).all();
					if (TIMING) {
						printf("eq: %d\n", eq.data()[0]);
					}
					sgx_time_t end = get_time();
					double elapsed = get_elapsed_time(start, end);
					if (TIMING) {
						printf("non-batched convd (%ld x %ld x %ld) took %.4f seconds\n", input.dimension(1), input.dimension(2), input.dimension(3), elapsed);
					}

					out1.resize(0);
					r.resize(0,0);
					b_r.resize(0);

					return output_map;
				}
				
			} else {	

				/************************************/	
				/*  VERIFICATION WITH PREPROCESSING */	
				/************************************/	

				if (r_left_data_ == NULL) {
					// pointwise convolution

					sgx_time_t start = get_time();
				
					// temporaries to store outputs
					res_x = mem_pool_->alloc<double>(h_out*w_out*REPS);
					res_z = mem_pool_->alloc<double>(h_out*w_out*REPS);
					TensorMap<double, 1> res_x_map(res_x, h_out*w_out*REPS);
					TensorMap<double, 1> res_z_map(res_z, h_out*w_out*REPS);
					res_z_map.setZero();

					// compute X*r_right + b*r_right
					preproc_verif_pointwise_X(input.data());

					if (release_input) {
						mem_pool_->release(input.data());				
					}

					// allocate memory for the presumably correct output
					T* output_mem_ = mem_pool_->alloc<T>(batch * output_size_);
					auto output_map = TensorMap<T, 4>(output_mem_, output_shape_);

					// compute Z*r_right and activation(Z) in a single loop
					if (activation_type_ == "relu") {
						preproc_verif_pointwise_Z_memfused(relu_avx, extra_data, output_map.data());
					} else if (activation_type_ == "relu6") {
						preproc_verif_pointwise_Z_memfused(relu6_avx, extra_data, output_map.data());
					} else {
						assert(activation_type_ == "linear" || activation_type_ == "softmax");
						preproc_verif_pointwise_Z_memfused(id_avx, extra_data, output_map.data());
					}

					// check equality mod p
					res_x_map = res_x_map - res_z_map;
					REDUCE_MOD_TENSOR(res_x_map);
					Tensor<bool, 0> eq = (res_x_map == static_cast<double>(0)).all();
					if (TIMING) {
						printf("eq: %d\n", eq.data()[0]);
					}
					sgx_time_t end = get_time();
					double elapsed = get_elapsed_time(start, end);
					if (TIMING) {
						printf("pointwise convd verif (%ld x %ld x %ld) took %.4f seconds\n", input.dimension(1), input.dimension(2), input.dimension(3), elapsed);
					}

					mem_pool_->release(res_x);
					mem_pool_->release(res_z);

					return output_map;
				}

				// sanity check
				assert(batch == 1);

				sgx_time_t start = get_time();

				// temporaries to store outputs
				res_x = mem_pool_->alloc<double>(REPS);
				res_z = mem_pool_->alloc<double>(REPS);
				TensorMap<double, 1> res_z_map(res_z, REPS);
				res_z_map.setZero();

				res_z_temp = mem_pool_->alloc<double>(REPS);
				TensorMap<double, 1> res_z_temp_map(res_z_temp, REPS);
				res_z_temp_map.setZero();

				// compute r_left * Conv(X, W*r_right) + r_left*bias*r_right
				preproc_verif_X(input.data());
				if (TIMING) {
					printf("r_inp_wr: %f, %f\n", mod_pos(res_x[0], p_verif), mod_pos(res_x[1], p_verif));
				}

				// allocate output mem
				if (release_input) {
					mem_pool_->release(input.data());
				}
				T* output_mem_ = mem_pool_->alloc<T>(batch * output_size_);
				auto output_map = TensorMap<T, 4>(output_mem_, output_shape_);

				// compute r_left*Z*r_right and activation(Z) in a single loop
				if (activation_type_ == "relu") {
					preproc_verif_Z_memfused([](__m256 x){return relu_avx(x);}, extra_data, output_map.data());
				} else if (activation_type_ == "relu6") { 
					preproc_verif_Z_memfused([](__m256 x){return relu6_avx(x);}, extra_data, output_map.data());
				} else {
					//assert(activation_type_ == "linear" || activation_type_ == "softmax");
					preproc_verif_Z_memfused([](__m256 x){return x;}, extra_data, output_map.data());
				}
				
				if (TIMING) {
					printf("r_out_r: %f, %f\n", mod_pos(res_z[0], p_verif), mod_pos(res_z[1], p_verif));
				}

				mem_pool_->release(res_x);
				mem_pool_->release(res_z);
				mem_pool_->release(res_z_temp);

				sgx_time_t end = get_time();
				double elapsed = get_elapsed_time(start, end);
				if (TIMING) {
					printf("convd verif preproc (%ld x %ld x %ld) took %.4f seconds\n",
							input.dimension(1), input.dimension(2), input.dimension(3), elapsed);
				}

				return output_map;
			}
        }

		T* kernel_data_;
		T* bias_data_;
		TensorMap<T, 4> kernel_;
		TensorMap<T, 1> bias_;

		const Eigen::PaddingType padding_;
		const int row_stride_;
		const int col_stride_;
		int pad_rows_;
		int pad_cols_;

		MemPool* mem_pool_;

		array4d input_shape_;
		array4d kernel_shape_;
		int input_size_;

		array4d output_shape_;
		int output_size_;

		bool lazy_load_;

		// for preprocessed verification
		bool verif_preproc_;

		TensorMap<double, 2> r_left_;
		TensorMap<double, 2> r_right_;
		TensorMap<T, 2> kernel_r_;
		TensorMap<double, 1> bias_r_;

		std::string activation_type_;
		MAC* mac;
	};

 const size_t kMaxChunkSize = (1 * 1024 * 1024);

 // adapted from tensorflow repository
 template <class T1, class T2, class T3>
 void conv2d_im2col(const T1* input_data,
          int input_batches, int input_height, int input_width,
          int input_depth, T2* filter_data, int filter_height,
          int filter_width, int filter_count, int stride_rows,
          int stride_cols, Eigen::PaddingType padding, T3* output_data,
          int output_height, int output_width, MemPool* mem_pool_) {


    // These calculations define how the patches will be positioned within the
    // input image. The actual definitions are quite complex, and rely on the
    // previously-calculated output size.
    int filter_left_offset;
    int filter_top_offset;
    if (padding == Eigen::PaddingType::PADDING_VALID) {
      filter_left_offset =
          ((output_width - 1) * stride_cols + filter_width - input_width + 1) /
          2;
      filter_top_offset = ((output_height - 1) * stride_rows + filter_height -
                           input_height + 1) /
                          2;
    } else {
      filter_left_offset =
          ((output_width - 1) * stride_cols + filter_width - input_width) / 2;
      filter_top_offset =
          ((output_height - 1) * stride_rows + filter_height - input_height) /
          2;
    }

    // The im2col buffer has # of patches rows, and # of filters cols.
    // It's laid out like this, in row major order in memory:
    //        < filter value count >
    //   ^   +---------------------+
    // patch |                     |
    // count |                     |
    //   v   +---------------------+
    // Each patch row contains a filter_width x filter_height patch of the
    // input, with the depth channel as the most contiguous in memory, followed
    // by the width, then the height. This is the standard memory order in the
    // image world if it helps to visualize it.
    const int filter_value_count = filter_width * filter_height * input_depth;
    assert((filter_value_count * sizeof(T1)) <= kMaxChunkSize);
    const int64 patches_per_chunk =
        kMaxChunkSize / (filter_value_count * sizeof(T1));
    const int64 chunk_value_count =
        (kMaxChunkSize + (sizeof(T1) - 1)) / sizeof(T1);

	// Because memory allocation is very expensive on mobile platforms, try to
    // allocate a persistent buffer that will be kept around between calls. We
    // use TensorFlow's resource management to ensure that the memory will be
    // released when the session is over.
	T1* im2col_buffer = mem_pool_->alloc<T1>(chunk_value_count);

    const int64 patch_count = (input_batches * output_height * output_width);
    const int64 chunk_count =
        (patch_count + (patches_per_chunk - 1)) / patches_per_chunk;
    for (int64 chunk_index = 0; chunk_index < chunk_count; ++chunk_index) {
      const int64 patch_index_start = chunk_index * patches_per_chunk;
      const int64 patch_index_end =
          std::min(patch_index_start + patches_per_chunk, patch_count);
      for (int64 patch_index = patch_index_start; patch_index < patch_index_end;
           ++patch_index) {
        const int64 batch = patch_index / (output_height * output_width);
        const int64 out_y = (patch_index / output_width) % output_height;
        const int64 out_x = patch_index % output_width;
        const T1* input_batch_start =
            input_data + (batch * input_height * input_width * input_depth);
        const int in_y_origin = (out_y * stride_rows) - filter_top_offset;
        const int in_x_origin = (out_x * stride_cols) - filter_left_offset;
        const int patch_index_within_chunk = patch_index % patches_per_chunk;
		T1* im2col_patch_start =
            im2col_buffer + (patch_index_within_chunk * filter_value_count);
        for (int filter_y = 0; filter_y < filter_height; ++filter_y) {
          const int in_y = in_y_origin + filter_y;
          T1* im2col_row_start =
              im2col_patch_start + (filter_y * filter_width * input_depth);
          // If we're off the top or the bottom of the input, fill the
          // whole row with zeroes.
          if ((in_y < 0) || (in_y >= input_height)) {
            T1* im2col_row_end =
                im2col_row_start + (filter_width * input_depth);
            std::fill(im2col_row_start, im2col_row_end, T1(0));
          } else {
            // What we're doing here is trying to copy and fill the im2col
            // buffer as efficiently as possible, using functions to set or
            // duplicate values en masse. We know we don't have to worry about
            // vertical edges because we dealt with that case above, so we
            // just need to handle filters that overlap the left or right
            // edges. Here's what that looks like:
            //
            // < left_zero_count > < center_copy_count > < right_zero_count >
            // +------------------+---------------------+--------------------+
            // |     (filter)     |       (image)       |      (filter)      |
            // +------------------+---------------------+--------------------+
            // in_x_origin        0                 input_width       in_x_end
            //
            // In reality it's unlikely that a filter patch will be wider
            // than an input, but this shows all the edge cases.
            // We use std::fill() to set the left and right sections to zeroes
            // and std::copy() to copy over the input data for the center.
            const int in_x_end = in_x_origin + filter_width;
            const int left_zero_count = std::max(0, 0 - in_x_origin);
            const int right_zero_count = std::max(0, in_x_end - input_width);
            const int center_copy_count =
                filter_width - (left_zero_count + right_zero_count);
            if (left_zero_count > 0) {
              T1* im2col_left_start = im2col_row_start;
              T1* im2col_left_end =
                  im2col_left_start + (left_zero_count * input_depth);
              std::fill(im2col_left_start, im2col_left_end, T1(0));
            }
            if (center_copy_count > 0) {
              const T1* input_row_start =
                  input_batch_start + (in_y * input_width * input_depth) +
                  (std::max(0, in_x_origin) * input_depth);
			  const T1* input_row_end =
                  input_row_start + (center_copy_count * input_depth);
              T1* im2col_center_start =
                  im2col_row_start + (left_zero_count * input_depth);
              std::copy(input_row_start, input_row_end, im2col_center_start);
            }
            if (right_zero_count > 0) {
              T1* im2col_right_start =
                  im2col_row_start +
                  ((left_zero_count + center_copy_count) * input_depth);
              T1* im2col_right_end =
                  im2col_right_start + (right_zero_count * input_depth);
              std::fill(im2col_right_start, im2col_right_end, T1(0));
            }
          }
        }
      }
      // Now we've assembled a set of image patches into a matrix, apply a
      // GEMM matrix multiply of the patches as rows, times the filter
      // weights in columns, to get partial results in the output matrix.
      const int how_many_patches = patch_index_end - patch_index_start;
      const int m = how_many_patches;
      const int n = filter_count;
      const int k = filter_value_count;
      const int lda = filter_value_count;
      const int ldb = filter_count;
      const int ldc = filter_count;
      T3* chunk_output_data = output_data + (patch_index_start * filter_count);

      Eigen::Map<Eigen::Matrix<T1, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> buffer_map(im2col_buffer, m, k);
      Eigen::Map<Eigen::Matrix<T2, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> kernel_map(filter_data, k, n);
      Eigen::Map<Eigen::Matrix<T3, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> output_map(chunk_output_data, m, n);
      output_map = buffer_map.template cast<T3>() * kernel_map.template cast<T3>();

   }
   mem_pool_->release(im2col_buffer);
 }
} //SGXDNN namespace

#endif
