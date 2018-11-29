#ifndef SGXDNN_GLOBAL_POOL_H
#define SGXDNN_GLOBAL_POOL_H

#include "assert.h"
#include <iostream>
#include <string>
#include "layer.hpp"

#ifdef USE_SGX
#include "Enclave.h"
#endif

using namespace tensorflow;

namespace SGXDNN
{

	template <typename T> class GlobalPool : public Layer<T>
	{
	public:
		explicit GlobalPool(
				const std::string& name,
				const array4d input_shape,
			    MemPool* mem_pool,
				bool verif_preproc
				): Layer<T>(name, input_shape),
				verif_preproc_(verif_preproc),
				mem_pool_(mem_pool)
		{
			output_shape_ = {1, 1, 1, input_shape[3]};
			output_size_ = input_shape[3];
		}

		array4d output_shape() override
		{
			return output_shape_;
		}

		int output_size() override
		{
			return output_size_;
		}

	protected:

		TensorMap<T, 4> apply_impl(TensorMap<T, 4> input, void* device_ptr = NULL, bool release_input = true) override
		{
			int batch = input.dimension(0);
            output_shape_[0] = batch;
            T* output_mem_ = mem_pool_->alloc<T>(batch * output_size_);
            auto output_map = TensorMap<T, 4>(output_mem_, output_shape_);

			Eigen::array<int, 2> mean_dims({1, 2});
			output_map = input.mean(mean_dims).reshape(output_shape_);

            mem_pool_->release(input.data());

            return output_map;
		}
		
		TensorMap<T, 4> fwd_verify_impl(TensorMap<T, 4> input, float** extra_data, int linear_idx, void* device_ptr = NULL, bool release_input = true) override
		{
			int batch = input.dimension(0);
            output_shape_[0] = batch;
            T* output_mem_ = mem_pool_->alloc<T>(batch * output_size_);
            auto output_map = TensorMap<T, 4>(output_mem_, output_shape_);

            Eigen::array<int, 2> mean_dims({1, 2});
            output_map = (input.mean(mean_dims).reshape(output_shape_)).round();

            mem_pool_->release(input.data());

            return output_map;
		}

		bool verif_preproc_;
		array4d output_shape_;
		int output_size_;
		MemPool* mem_pool_;
	};
}
#endif
