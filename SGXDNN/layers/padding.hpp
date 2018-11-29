#ifndef SGXDNN_PADDING2D_H_
#define SGXDNN_PADDING2D_H_

#include "../mempool.hpp"
#include "layer.hpp"

using namespace tensorflow;

namespace SGXDNN
{
	template <typename T> class Padding2D: public Layer<T>
    {
    public:
        explicit Padding2D(
                const std::string& name,
                const array4d input_shape,
                const int pad_rows,
                const int pad_cols,
                MemPool* mem_pool
                ): Layer<T>(name, input_shape),
                pad_rows_(pad_rows),
                pad_cols_(pad_cols),
                mem_pool_(mem_pool)
        {
            int input_rows_ = input_shape[1];
            int input_cols_ = input_shape[2];
            int input_depth_ = input_shape[3];

			out_rows_ = input_rows_ + 2*pad_rows_;
			out_cols_ = input_cols_ + 2*pad_cols_;
            output_shape_ = {0, out_rows_, out_cols_, input_depth_};
            output_size_ = out_rows_ * out_cols_ * input_depth_;

            printf("in Padding2D with input_shape = (%d, %d, %d), pad = (%d, %d), out_shape = (%d, %d, %d)\n",
                   input_rows_, input_cols_, input_depth_, pad_rows_, pad_cols_, out_rows_, out_cols_, input_depth_);

			paddings_[0] = std::make_pair(0, 0);
			paddings_[1] = std::make_pair(pad_rows_, pad_rows_);
			paddings_[2] = std::make_pair(pad_cols_, pad_cols_);
			paddings_[3] = std::make_pair(0, 0);
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

			output_map = input.pad(paddings_);	

            mem_pool_->release(input.data());

            return output_map;
        }

        int out_rows_;
        int out_cols_;
        int pad_rows_;
        int pad_cols_;
		Eigen::array<std::pair<int, int>, 4> paddings_;
		MemPool* mem_pool_;

		array4d output_shape_;
        int output_size_;
	};
}
#endif
