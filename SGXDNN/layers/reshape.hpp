#ifndef SGXDNN_RESHAPE_H
#define SGXDNN_RESHAPE_H

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

	template <typename T> class Reshape: public Layer<T>
	{
	public:
		explicit Reshape(
				const std::string& name,
				const array4d input_shape,
				const array3d target_shape
				): Layer<T>(name, input_shape)
		{
			output_shape_ = {1, std::max((int) target_shape[0], 1), std::max((int) target_shape[1], 1), std::max((int) target_shape[2], 1)};
			output_size_ = target_shape[0] * target_shape[1] * target_shape[2];
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
       	    auto output_map = TensorMap<T, 4>(input.data(), output_shape_);

			//printf("reshaping from (%ld, %ld, %ld, %ld) to (%ld, %ld, %ld, %ld)\n",
			//	   input.dimension(0), input.dimension(1), input.dimension(2), input.dimension(3),
			//	   output_map.dimension(0), output_map.dimension(1), output_map.dimension(2), output_map.dimension(3));
            return output_map;
		}
		
		array4d output_shape_;
		int output_size_;
	};
}
#endif
