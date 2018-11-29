#ifndef SGXDNN_MAXPOOL2D_H_
#define SGXDNN_MAXPOOL2D_H_

#include <iostream>
#include <string>

#include "../mempool.hpp"
#include "layer.hpp"
#include "eigen_maxpool.h"

using namespace tensorflow;

namespace SGXDNN
{

	template<typename T>
	void fast_maxpool(T* input, T* output, 
					  int batch, int input_rows_, int input_cols_, int input_depth_, int out_rows_, int out_cols_,
					  int window_rows_, int window_cols_, int pad_rows_, int pad_cols_, int row_stride_, int col_stride_,
					  bool avg_pool = false)
	{
		typedef Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>> ConstEigenMatrixMap;
		typedef Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>> EigenMatrixMap;

		ConstEigenMatrixMap in_mat(input, input_depth_,
								   input_cols_ * input_rows_ * batch);

		EigenMatrixMap out_mat(output, input_depth_, out_rows_ * out_cols_ * batch);

		// The following code basically does the following:
		// 1. Flattens the input and output tensors into two dimensional arrays.
		//    tensor_in_as_matrix:
		//      depth by (tensor_in_cols * tensor_in_rows * tensor_in_batch)
		//    output_as_matrix:
		//      depth by (out_width * out_height * tensor_in_batch)
		//
		// 2. Walks through the set of columns in the flattened
		// tensor_in_as_matrix,
		//    and updates the corresponding column(s) in output_as_matrix with the
		//    max value.
		auto shard = [&in_mat, &out_mat, input_rows_, input_cols_, input_depth_, out_rows_, out_cols_,
					  window_rows_, window_cols_, pad_rows_, pad_cols_, row_stride_, col_stride_, avg_pool](long start, long limit) {
			const int in_rows = input_rows_;
			const int in_cols = input_cols_;
			const int window_rows = window_rows_;
			const int window_cols = window_cols_;
			const int pad_rows = pad_rows_;
			const int pad_cols = pad_cols_;
			const int row_stride = row_stride_;
			const int col_stride = col_stride_;
			const int out_height = out_rows_;
			const int out_width = out_cols_;
			const int input_depth = input_depth_;
			 
			{
		  		// Initializes the output tensor with MIN<T>.
		  		const int output_image_size = out_height * out_width * input_depth;
		  		EigenMatrixMap out_shard(out_mat.data() + start * output_image_size,
								   1, (limit - start) * output_image_size);

				if (avg_pool) {
		  			out_shard.setConstant((T) 0.0);
				} else {
		  			out_shard.setConstant(Eigen::NumTraits<T>::lowest());
				}
			}

			for (int b = start; b < limit; ++b) {
			    const int out_offset_batch = b * out_height;
			    for (int h = 0; h < in_rows; ++h) {
				  for (int w = 0; w < in_cols; ++w) {
		  	  	    // (h_start, h_end) * (w_start, w_end) is the range that the input
			  	    // vector projects to.
			  	    const int hpad = h + pad_rows;
			  	    const int wpad = w + pad_cols;
			  	    const int h_start = (hpad < window_rows)
											? 0
											: (hpad - window_rows) / row_stride + 1;
			  	    const int h_end = std::min(hpad / row_stride + 1, out_height);
			  	    const int w_start = (wpad < window_cols)
											? 0
											: (wpad - window_cols) / col_stride + 1;
			  	    const int w_end = std::min(wpad / col_stride + 1, out_width);
			  	    // compute elementwise max
			  	    const int in_offset = (b * in_rows + h) * in_cols + w;
			  	    for (int ph = h_start; ph < h_end; ++ph) {
					  const int out_offset_base =
					  	  (out_offset_batch + ph) * out_width;
					  for (int pw = w_start; pw < w_end; ++pw) {
				  	    const int out_offset = out_offset_base + pw;
						if (avg_pool) {
							out_mat.col(out_offset) += in_mat.col(in_offset) / ((T)(window_rows * window_cols));
						} else {
				  	    	out_mat.col(out_offset) = out_mat.col(out_offset).cwiseMax(in_mat.col(in_offset));
						}
					  }
			  	    }
				  }
			    }
			}
		};

		shard(0, batch);
	}


	template <typename T> class MaxPool2D : public Layer<T>
	{
	public:
		explicit MaxPool2D(
                const std::string& name,
				const array4d input_shape,
                const int window_rows,
                const int window_cols,
                const int row_stride,
                const int col_stride,
                const Eigen::PaddingType& padding,
				const bool avg_pool,
				MemPool* mem_pool
                ): Layer<T>(name, input_shape),
                window_rows_(window_rows),
                window_cols_(window_cols),
                row_stride_(row_stride),
                col_stride_(col_stride),
                padding_(padding),
				avg_pool_(avg_pool),
				mem_pool_(mem_pool)
		{
			input_rows_ = input_shape[1];
			input_cols_ = input_shape[2];
			input_depth_ = input_shape[3];

			GetWindowedOutputSize(input_rows_, window_rows_, row_stride_,
								  padding_, &out_rows_, &pad_rows_);
			GetWindowedOutputSize(input_cols_, window_cols_, col_stride_,
								  padding_, &out_cols_, &pad_cols_);

			output_shape_ = {0, out_rows_, out_cols_, input_depth_};
			output_size_ = out_rows_ * out_cols_ * input_depth_;

			printf("in Pool2D with window = (%d, %d), stride = (%d, %d), padding = %d, out_shape = (%d, %d, %d), pad = (%d, %d)\n",
			        window_rows_, window_cols_, row_stride_, col_stride_, padding_, out_rows_, out_cols_, input_depth_, pad_rows_, pad_cols_);
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

			fast_maxpool(input.data(), output_map.data(),
                         batch, input_rows_, input_cols_, input_depth_, out_rows_, out_cols_,
                      	 window_rows_, window_cols_, pad_rows_, pad_cols_, row_stride_, col_stride_, avg_pool_);

			mem_pool_->release(input.data());
			return output_map;
		}

		TensorMap<T, 4> fwd_verify_impl(TensorMap<T, 4> input, float** aux_data, int linear_idx, void* device_ptr = NULL, bool release_input = true) override
        {
            auto output_map = apply_impl(input, device_ptr, release_input);

            if (avg_pool_) {
                output_map = output_map.round();
            }

            return output_map;
        }


		int input_rows_;
		int input_cols_;
		int input_depth_;

		int out_rows_;
		int out_cols_;
		int pad_rows_;
		int pad_cols_;	

		const Eigen::PaddingType padding_;
		const int window_rows_;
		const int window_cols_;
		const int row_stride_;
		const int col_stride_;
		const bool avg_pool_;
		MemPool* mem_pool_;

		array4d output_shape_;
		int output_size_;
	};
}
#endif
