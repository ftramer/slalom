#ifndef SGXDNN_BLOCK_H_
#define SGXDNN_BLOCK_H_

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

	template <typename T> class ResNetBlock: public Layer<T>
	{
	public:
		explicit ResNetBlock(
				const std::string& name,
				const array4d input_shape,
				bool identity,
				std::vector<std::shared_ptr<Layer<T>>> path1,
				std::vector<std::shared_ptr<Layer<T>>> path2,
			    int bits_w,
                int bits_x,
				MemPool* mem_pool
				): Layer<T>(name, input_shape),
				identity_(identity),
				path1_(path1),
				path2_(path2),
				bits_w_(bits_w),
                bits_x_(bits_x),
				mem_pool_(mem_pool)
		{
			Conv2D<T>* final_conv = dynamic_cast<Conv2D<T>*>(path1[path1.size() - 1].get());
			output_shape_ = final_conv->output_shape();
			output_size_ = final_conv->output_size();
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
			int count = 0;

			for (int i=0; i<path1_.size(); i++) {
				count += path1_[i]->num_linear();
			}

			for (int i=0; i<path2_.size(); i++) {
				count += path2_[i]->num_linear();
            }
			return count;
		}

		std::vector<std::shared_ptr<Layer<T>>> get_path1()
		{
			return path1_;
		}

		std::vector<std::shared_ptr<Layer<T>>> get_path2()
		{
			return path2_;
		}

	protected:

		TensorMap<T, 4> apply_impl(TensorMap<T, 4> input, void* device_ptr = NULL, bool release_input = true) override
		{
			TensorMap<float, 4>* in_ptr1 = &input;

			// loop over first path
			for (int i=0; i<path1_.size(); i++) {
				if (TIMING) {
					printf("\tbefore layer %d (%s)\n", i, path1_[i]->name_.c_str());
				}

				sgx_time_t layer_start = get_time();
				bool release_input = i>0;
				auto temp_output = path1_[i]->apply(*in_ptr1, device_ptr, release_input);

				in_ptr1 = &temp_output;

				sgx_time_t layer_end = get_time();
				if (TIMING) {
					printf("\tlayer %d required %4.4f sec\n", i, get_elapsed_time(layer_start, layer_end));
				}
			}

			TensorMap<float, 4>* in_ptr2 = &input;

            // loop over second path
            for (int i=0; i<path2_.size(); i++) {
                if (TIMING) {
                    printf("\tbefore layer %d (%s)\n", i, path2_[i]->name_.c_str());
                }

                sgx_time_t layer_start = get_time();
                auto temp_output = path2_[i]->apply(*in_ptr2, device_ptr);

                in_ptr2 = &temp_output;

                sgx_time_t layer_end = get_time();
                if (TIMING) {
                    printf("\tlayer %d required %4.4f sec\n", i, get_elapsed_time(layer_start, layer_end));
                }
            }

			TensorMap<float, 4> out1 = *in_ptr1;
			TensorMap<float, 4> out2 = *in_ptr2;

			out1 = (out1 + out2).cwiseMax(static_cast<T>(0));
			mem_pool_->release(out2.data());

			return out1;
		}

		TensorMap<T, 4> fwd_verify_impl(TensorMap<T, 4> input, float** aux_data, int linear_idx, void* device_ptr = NULL, bool release_input = true) override
		{
			TensorMap<float, 4>* in_ptr1 = &input;

			for (int i=0; i<path1_.size(); i++) {
				if (TIMING) {
					printf("\tbefore layer %d (%s)\n", i, path1_[i]->name_.c_str());
				}

				sgx_time_t layer_start = get_time();
				bool release_input = i>0;
				auto temp_output = path1_[i]->fwd_verify(*in_ptr1, aux_data, linear_idx, device_ptr, release_input);

				in_ptr1 = &temp_output;

				linear_idx += path1_[i]->num_linear();

				sgx_time_t layer_end = get_time();
				if (TIMING) {
					printf("\tlayer %d required %4.4f sec\n", i, get_elapsed_time(layer_start, layer_end));
				}
			}

			TensorMap<float, 4>* in_ptr2 = &input;
			for (int i=0; i<path2_.size(); i++) {
                if (TIMING) {
                    printf("\tbefore layer %d (%s)\n", i, path2_[i]->name_.c_str());
                }

                sgx_time_t layer_start = get_time();
                auto temp_output = path2_[i]->fwd_verify(*in_ptr2, aux_data, linear_idx, device_ptr);

                in_ptr2 = &temp_output;

                linear_idx += path2_[i]->num_linear();

                sgx_time_t layer_end = get_time();
                if (TIMING) {
                    printf("\tlayer %d required %4.4f sec\n", i, get_elapsed_time(layer_start, layer_end));
                }
            }

			TensorMap<float, 4> out1 = *in_ptr1;
            TensorMap<float, 4> out2 = *in_ptr2;

			int shift_w = (1 << bits_w_);
			T shift = 1.0/shift_w;
			if (identity_) {
            	out1 = ((out1 + out2 * ((T)shift_w)).cwiseMax(static_cast<T>(0)) * shift).round();
			} else {
            	out1 = ((out1 + out2).cwiseMax(static_cast<T>(0)) * shift).round();
			}

            mem_pool_->release(out2.data());

            return out1;
		}

		array4d output_shape_;
		int output_size_;
		bool identity_;
		const int bits_w_;
        const int bits_x_;
		MemPool* mem_pool_;

		std::vector<std::shared_ptr<Layer<T>>> path1_;
		std::vector<std::shared_ptr<Layer<T>>> path2_;
	};
}
#endif
