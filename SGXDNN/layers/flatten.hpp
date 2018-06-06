#ifndef SGXDNN_FLATTEN_H_
#define SGXDNN_FLATTEN_H_

#include "assert.h"
#include <iostream>
#include <string>

#include "layer.hpp"

using namespace tensorflow;

namespace SGXDNN
{
	template <typename T> class Flatten : public Layer<T>
	{
	public:
		explicit Flatten(const std::string& name,
						 array4d input_shape):
		Layer<T>(name, input_shape)
		{
			const int input_rows = input_shape[1];
			const int input_cols = input_shape[2];
			const int input_depth = input_shape[3];

			output_shape_ = {1, 1, 0, input_rows * input_cols * input_depth};
			output_size_ = input_rows * input_cols * input_depth;
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

		TensorMap<T, 4> apply_impl(TensorMap<T, 4> input) override
		{
			const int batch = input.dimension(0);
			output_shape_[2] = batch;
			return input.reshape(output_shape_);;
		}

		array4d output_shape_;
		int output_size_;
	};
}

#endif
