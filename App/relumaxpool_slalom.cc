#define EIGEN_USE_THREADS

#include "tensorflow/core/framework/numeric_op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/util/padding.h"
#include "tensorflow/core/util/tensor_format.h"
#include "tensorflow/core/framework/common_shape_fns.h"

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include <dlfcn.h>

namespace tensorflow {

Eigen::PaddingType BrainPadding2EigenPadding(Padding padding) {
  switch (padding) {
    case Padding::VALID:
      return Eigen::PADDING_VALID;
    case Padding::SAME:
      return Eigen::PADDING_SAME;
  }
  return Eigen::PADDING_SAME;  // Prevent compiler warning about missing return
}

// A helper class to manage sizes and shapes for pooling operations.
struct PoolParameters {
  // Updates context->status if there is an invalid input.
  PoolParameters(OpKernelContext* context, const std::vector<int32>& ksize,
                 const std::vector<int32>& stride, Padding padding,
                 TensorFormat data_format, const TensorShape& tensor_in_shape);

  // Returns the shape of the output for "forward" pooling operations.
  TensorShape forward_output_shape();

  int depth;

  int tensor_in_cols;
  int tensor_in_rows;
  int tensor_in_batch;

  int window_rows;
  int window_cols;
  int depth_window;

  int row_stride;
  int col_stride;
  int depth_stride;

  int64 out_height;
  int64 out_width;
  int out_depth;

  int64 pad_rows;
  int64 pad_cols;
  int pad_depth;

  TensorFormat data_format;
};


PoolParameters::PoolParameters(OpKernelContext* context,
                               const std::vector<int32>& ksize,
                               const std::vector<int32>& stride,
                               Padding padding, TensorFormat data_format,
                               const TensorShape& tensor_in_shape) {
  // For maxpooling, tensor_in should have 2 spatial dimensions.
  // Note: the total number of dimensions could be 4 for NHWC, NCHW,
  // or 5 for NCHW_VECT_C.
  OP_REQUIRES(context,
              GetTensorSpatialDims(tensor_in_shape.dims(), data_format) == 2,
              errors::InvalidArgument(
                  "tensor_in_shape must have 2 spatial dimensions. ",
                  tensor_in_shape.dims(), " ", data_format));

  this->data_format = data_format;
  depth = GetTensorDim(tensor_in_shape, data_format, 'C') *
          (data_format == FORMAT_NCHW_VECT_C ? 4 : 1);
  tensor_in_cols = GetTensorDim(tensor_in_shape, data_format, 'W');
  tensor_in_rows = GetTensorDim(tensor_in_shape, data_format, 'H');
  tensor_in_batch = GetTensorDim(tensor_in_shape, data_format, 'N');
  window_rows = GetTensorDim(ksize, data_format, 'H');
  window_cols = GetTensorDim(ksize, data_format, 'W');
  depth_window = GetTensorDim(ksize, data_format, 'C');
  row_stride = GetTensorDim(stride, data_format, 'H');
  col_stride = GetTensorDim(stride, data_format, 'W');
  depth_stride = GetTensorDim(stride, data_format, 'C');

  // We only support 2D pooling across width/height and depthwise
  // pooling, not a combination.
  OP_REQUIRES(context,
              (depth_window == 1 || (window_rows == 1 && window_cols == 1)),
              errors::Unimplemented(
                  "MaxPooling supports exactly one of pooling across depth "
                  "or pooling across width/height."));

  if (depth_window == 1) {
    OP_REQUIRES_OK(
        context, GetWindowedOutputSize(tensor_in_rows, window_rows, row_stride,
                                       padding, &out_height, &pad_rows));
    OP_REQUIRES_OK(
        context, GetWindowedOutputSize(tensor_in_cols, window_cols, col_stride,
                                       padding, &out_width, &pad_cols));
    pad_depth = 0;
    out_depth = depth;
  }
}

TensorShape PoolParameters::forward_output_shape() {
  if (depth_window == 1) {
    // Spatial pooling
    return ShapeFromFormat(data_format, tensor_in_batch, out_height, out_width,
                           depth);
  } else {
    // Depthwise pooling
    return TensorShape(
        {tensor_in_batch, tensor_in_rows, tensor_in_cols, out_depth});
  }
}


template <typename Device, typename T>
class ReluMaxPoolSlalomOp : public OpKernel {
 public:
  explicit ReluMaxPoolSlalomOp(OpKernelConstruction* context) : OpKernel(context) {
    string data_format;
    auto status = context->GetAttr("data_format", &data_format);
    if (status.ok()) {
      OP_REQUIRES(context, FormatFromString(data_format, &data_format_),
                  errors::InvalidArgument("Invalid data format"));
      OP_REQUIRES(
          context, data_format_ == FORMAT_NHWC,
          errors::InvalidArgument("Default MaxPoolingOp only supports NHWC."));
    } else {
      data_format_ = FORMAT_NHWC;
    }
    OP_REQUIRES_OK(context, context->GetAttr("ksize", &ksize_));
    OP_REQUIRES(context, ksize_.size() == 4,
                errors::InvalidArgument("Sliding window ksize field must "
                                        "specify 4 dimensions"));
    OP_REQUIRES_OK(context, context->GetAttr("strides", &stride_));
    OP_REQUIRES(context, stride_.size() == 4,
                errors::InvalidArgument("Sliding window stride field must "
                                        "specify 4 dimensions"));
    OP_REQUIRES_OK(context, context->GetAttr("padding", &padding_));
    OP_REQUIRES(context, ksize_[0] == 1 && stride_[0] == 1,
                errors::Unimplemented(
                    "Pooling is not yet supported on the batch dimension."));

#ifdef USE_SGX
    OP_REQUIRES_OK(context, context->GetAttr("eid_low", &eid_low_));
    OP_REQUIRES_OK(context, context->GetAttr("eid_high", &eid_high_));
	lib_ = dlopen("App/enclave_bridge.so", RTLD_NOW);
#else
	lib_ = dlopen("lib/sgxdnn.so", RTLD_NOW);
#endif
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& tensor_in = context->input(0);
    const Tensor& blind = context->input(1);

	//std::cout << "in relumaxpool!" << std::endl;    
    //std::cout << "input: " << tensor_in.DebugString() << std::endl;
    //std::cout << "blind: " << blind.DebugString() << std::endl;

    PoolParameters params{context,  ksize_,      stride_,
                          padding_, FORMAT_NHWC, tensor_in.shape()};
    if (!context->status().ok()) {
      return;
    }

    Tensor* output = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(
                                0, params.forward_output_shape(), &output));

    SpatialMaxPool(context, output, tensor_in, blind, params, padding_);
  }

 private:

  void SpatialMaxPool(OpKernelContext* context, Tensor* output,
                      const Tensor& input, const Tensor& blind,
		      		  const PoolParameters& params,
                      const Padding& padding) {

    Eigen::PaddingType pt = BrainPadding2EigenPadding(padding);

    auto dim_in_ = input.tensor<T, 4>().dimensions();
    long int dim_in[4] = {dim_in_[0], dim_in_[1], dim_in_[2], dim_in_[3]};

    auto dim_out_ = output->tensor<T, 4>().dimensions();
    long int dim_out[4] = {dim_out_[0], dim_out_[1], dim_out_[2], dim_out_[3]};

#ifdef USE_SGX
    unsigned long int eid_ = (eid_high_ << 32) + eid_low_;
	typedef void (*maxpoolrelu_ecall)(unsigned long int eid, float* in, float* out, float* blind, 
								  	  long int dim_in[4], long int dim_out[4], 
								  	  int window_rows, int window_cols, int row_stride, int col_stride, bool same_padding);
    dlerror();

    maxpoolrelu_ecall mpr = (maxpoolrelu_ecall) dlsym(lib_, "slalom_maxpoolrelu");
    const char *dlsym_error = dlerror();
    OP_REQUIRES(context, !dlsym_error, errors::Unknown("loading of maxpoolrelu failed: ", dlsym_error));

    mpr(eid_, (float*) input.flat<T>().data(), (float*) output->flat<T>().data(), (float*) blind.flat<T>().data(),
	    dim_in, dim_out, params.window_rows, params.window_cols, params.row_stride, params.col_stride, (pt == Eigen::PaddingType::PADDING_SAME));

#else

	typedef void (*maxpoolrelu_ecall)(float* in, float* out, float* blind,
                                  	  long int dim_in[4], long int dim_out[4], 
                                  	  int window_rows, int window_cols, int row_stride, int col_stride, bool same_padding);
    dlerror();

    maxpoolrelu_ecall mpr = (maxpoolrelu_ecall) dlsym(lib_, "slalom_maxpoolrelu");
    const char *dlsym_error = dlerror();
    OP_REQUIRES(context, !dlsym_error, errors::Unknown("loading of maxpoolrelu failed: ", dlsym_error));

    mpr((float*) input.flat<T>().data(), (float*) output->flat<T>().data(), (float*) blind.flat<T>().data(),
        dim_in, dim_out, params.window_rows, params.window_cols, params.row_stride, params.col_stride, (pt == Eigen::PaddingType::PADDING_SAME));

#endif
  }

  std::vector<int32> ksize_;
  std::vector<int32> stride_;
  Padding padding_;
  TensorFormat data_format_;
  void* lib_;

#ifdef USE_SGX
  int64 eid_low_;
  int64 eid_high_;
#endif
};

typedef Eigen::ThreadPoolDevice CPUDevice;

REGISTER_KERNEL_BUILDER(Name("ReluMaxPoolSlalom").Device(DEVICE_CPU), ReluMaxPoolSlalomOp<CPUDevice, float>);

REGISTER_OP("ReluMaxPoolSlalom")
    .Attr(
        "T: {half, bfloat16, float, double, int32, int64, uint8, int16, int8, "
        "uint16, qint8} = DT_FLOAT")
    .Attr("ksize: list(int) >= 4")
    .Attr("strides: list(int) >= 4")
    .Attr(GetPaddingAttrString())
    .Attr("data_format: {'NHWC', 'NCHW', 'NCHW_VECT_C'} = 'NHWC'")
#ifdef USE_SGX
    .Attr("eid_low: int")
    .Attr("eid_high: int")
#endif
    .Input("inp: float")
    .Input("blind: float")
    .Output("output: T")
    .SetShapeFn(shape_inference::MaxPoolShape);

} // namespace tensorflow

#undef EIGEN_USE_THREADS
