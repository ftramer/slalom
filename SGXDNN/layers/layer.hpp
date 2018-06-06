#ifndef SGXDNN_LAYER_H_
#define SGXDNN_LAYER_H_

#define USE_EIGEN_TENSOR

#include <string>
#include <unsupported/Eigen/CXX11/Tensor>

#include "tensor_types.h"
#include "shape.h"
#include "../utils.hpp"

using namespace tensorflow;

namespace SGXDNN
{
    template <typename T>
    class Layer {

    public:
        explicit Layer(const std::string& name,
        			   const array4d input_shape)
            : name_(name),
			  input_shape_(input_shape)
        {
        }

        virtual ~Layer()
        {
        }

        TensorMap<T, 4> apply(TensorMap<T, 4> input_map, void* device_ptr = NULL)  {
            auto result = apply_impl(input_map, device_ptr);
            return result;
        }

        TensorMap<T, 4> fwd_verify(TensorMap<T, 4> input_map, float* extra_data, void* device_ptr = NULL)  {
            auto result = fwd_verify_impl(input_map, extra_data, device_ptr);
            return result;
        }

        bool batch_verify(float* input_data, float* output_data, int batch_size)  {
			return batch_verify_impl(input_data, output_data, batch_size);
		}

        virtual array4d output_shape() = 0;
        virtual int output_size() = 0;

		virtual bool is_linear() {
			return false;
		}

        std::string name_;
        const array4d input_shape_;

    protected:
        virtual TensorMap<T, 4> apply_impl(TensorMap<T, 4> input_map, void* device_ptr = NULL) = 0;

        virtual TensorMap<T, 4> fwd_verify_impl(TensorMap<T, 4> input_map, float* extra_data, void* device_ptr = NULL) {
			return apply_impl(input_map, device_ptr);
		}

        virtual bool batch_verify_impl(float* input_data, float* extra_data, int batch_size) {
			return true;
		}
    };
}
#endif
