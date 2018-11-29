#ifndef SGXDNN_DENSE_H_
#define SGXDNN_DENSE_H_

#include <stdio.h>
#include <iostream>
#include <string>

#include "../mempool.hpp"
#include "layer.hpp"

#include "../Crypto.h"

#ifdef USE_SGX
#include "Enclave.h"
#endif

using namespace tensorflow;

namespace SGXDNN
{

	template <typename T> class Dense : public Layer<T>
	{
	public:
		explicit Dense(
				const std::string& name,
				const array4d input_shape,
				const int h_in,
				const int h_out,
				T* kernel, T* bias,
				MemPool* mem_pool,
				bool is_verif_mode,
				bool verif_preproc
				): Layer<T>(name, input_shape),
				h_in_(h_in),
				h_out_(h_out),
				kernel_data_(nullptr),
			    bias_data_(nullptr),
				r_right_data_(nullptr),
				kernel_r_data_(nullptr),
				bias_r_data_(nullptr),
				kernel_(NULL, h_in, h_out),
				bias_(NULL, h_out),
				r_right_(NULL, REPS, h_out),
				bias_r_(NULL, REPS),
				kernel_r_(NULL, h_in, REPS),
				mem_pool_(mem_pool),
				output_mem_(nullptr),
				verif_preproc_(verif_preproc),
				mac(nullptr)
		{
			assert(!verif_preproc);

			use_sharding_ = false;
			#ifdef USE_SGX
			// shard if weight matrix is bigger than 8MB
			if (h_in * h_out > 2 * 1000 * 1000) {
				use_sharding_ = true;
				mac = new MAC();

				// TODO compute original MAC tags
			}
			#endif

			if (use_sharding_) {
				// keep the weights outside the enclave for now
				kernel_data_ = kernel;
			} else {
				long kernel_size = h_in * h_out;
				kernel_data_ = mem_pool_->alloc<T>(kernel_size);
				std::copy(kernel, kernel + kernel_size, kernel_data_);
				new (&kernel_) MatrixMap<T>(kernel_data_, h_in, h_out);
			}

			long bias_size = h_out;
			bias_data_ = mem_pool_->alloc<T>(bias_size);
			std::copy(bias, bias + bias_size, bias_data_);
			new (&bias_) typename TTypes<T>::ConstVec(bias_data_, h_out);

			output_shape_ = {1, 1, 0, h_out};
			output_size_ = h_out;
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

	protected:

		TensorMap<T, 4> apply_impl(TensorMap<T, 4> input, void* device_ptr = NULL, bool release_input = true) override
		{

			int batch;

			// flatten input if necessary
			if (input.dimension(0) == 1 && input.dimension(1) == 1)
			{
				batch = input.dimension(2);
			}
			else
			{
				batch = input.dimension(0);
			}
			output_shape_[2] = batch;

			output_mem_ = mem_pool_->alloc<T>(batch * output_size_);
			auto output_map = TensorMap<T, 4>(output_mem_, output_shape_);

			sgx_time_t start = get_time();

			if (use_sharding_) {
				int shard_factor = 16;
				int sharded_h_in = h_in_ / shard_factor;
				T* kernel_shard = mem_pool_->alloc<T>(sharded_h_in * h_out_);

				for (int i=0; i<shard_factor; i++) {
					// read some rows of the kernel
					std::copy(kernel_data_ + i * sharded_h_in * h_out_, kernel_data_ + (i+1) * sharded_h_in * h_out_, kernel_shard);
					new (&kernel_) Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
							(kernel_shard, sharded_h_in, h_out_);

					// TODO actually check mac...
					Tag tag = mac->mac((uint8_t*) kernel_shard, sharded_h_in * h_out_ * sizeof(T));

					if (batch == 1) {
						auto output_matrix_map = VectorMap<T>(output_mem_, h_out_);
						auto input_matrix_map = VectorMap<T>(input.data() + i*sharded_h_in, sharded_h_in);
						if (i == 0) {
                        	output_matrix_map.noalias() = input_matrix_map * kernel_;
                    	} else {
                        	output_matrix_map.noalias() = output_matrix_map + input_matrix_map * kernel_;
                    	}
					}
					else
					{
						for(int b=0; b<batch; b++) {
							auto output_matrix_map = VectorMap<T>(output_mem_ + b*h_out_, h_out_);
							auto input_matrix_map = VectorMap<T>(input.data() + b*h_in_ + i*sharded_h_in, sharded_h_in);

							if (i == 0) {
								output_matrix_map.noalias() = input_matrix_map * kernel_;
							} else {
								output_matrix_map.noalias() = output_matrix_map + input_matrix_map * kernel_;
							}
						}
					}
				}

				mem_pool_->release(kernel_shard);
			}
			else
			{
				if (batch == 1) {
					VectorMap<T> output_matrix_map(output_mem_, h_out_);
					VectorMap<T> input_matrix_map(input.data(), h_in_);
					output_matrix_map = input_matrix_map * kernel_;
				}
				else
				{
					auto output_matrix_map = MatrixMap<T>(output_mem_, batch, h_out_);
					auto input_matrix_map = MatrixMap<T>(input.data(), batch, h_in_);
					output_matrix_map.noalias() = input_matrix_map * kernel_;
				}
			}

			sgx_time_t end = get_time();
			double elapsed = get_elapsed_time(start, end);
			if (TIMING) {
				printf("dense (%d x %d) took %.4f seconds\n", h_in_, h_out_, elapsed);
			}

			const int bias_size = bias_.dimension(0);
            const int rest_size = output_map.size() / bias_size;
            array1d one_d = {output_map.size()};
            array1d bcast = {rest_size};
			output_map.reshape(one_d) = output_map.reshape(one_d) + bias_.broadcast(bcast).reshape(one_d);

			mem_pool_->release(input.data());
			return output_map;
		}

		TensorMap<T, 4> fwd_verify_impl(TensorMap<T, 4> input, float** extra_data, int linear_idx, void* device_ptr, bool release_input = true) {
			int batch;
			assert(-1);

			if (input.dimension(0) == 1 && input.dimension(1) == 1)
			{
				batch = input.dimension(2);
			}
			else
			{
				batch = input.dimension(0);
			}
			output_shape_[2] = batch;

			if (verif_preproc_) {
				assert(batch == 1);
			}
			return apply_impl(input, device_ptr);
		}

		const int h_in_;
		const int h_out_;
		T* kernel_data_;
		T* bias_data_;
		MatrixMap<T> kernel_;
		TensorMap<T, 1> bias_;
		MemPool* mem_pool_;
		T* output_mem_;
		bool use_sharding_;

		array4d output_shape_;
		int output_size_;

		// verify with preprocessing
		bool verif_preproc_;
        double* r_right_data_;
        double* kernel_r_data_;
        double* bias_r_data_;

        TensorMap<double, 2> r_right_;
        TensorMap<double, 2> kernel_r_;
        TensorMap<double, 1> bias_r_;

        std::string activation_type_;

        MAC* mac;
	};
}

#endif
