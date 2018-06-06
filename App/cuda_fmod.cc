#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"

using namespace tensorflow;
using GPUDevice = Eigen::GpuDevice;

REGISTER_OP("ModCast")
	.Attr("q: float")
	.Attr("p: float")
    .Input("inp_low: float")
    .Input("inp_high: float")
    .Output("output: float")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(0));
      return Status::OK();
    });

void ModCastFunctor(int size, const float* in_low, const float* in_high, const double q, const double p, float* out);

//template <typename Device>
class ModCastOp : public OpKernel {
 public:
  explicit ModCastOp(OpKernelConstruction* context) : OpKernel(context) {
	OP_REQUIRES_OK(context, context->GetAttr("q", &q_));
	OP_REQUIRES_OK(context, context->GetAttr("p", &p_));
  }

  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    const Tensor& input_low_tensor = context->input(0);
    const Tensor& input_high_tensor = context->input(1);

    // Create an output tensor
    Tensor* output_tensor = NULL;
	OP_REQUIRES_OK(context, context->forward_input_or_allocate_output(
                                {0}, 0, input_low_tensor.shape(), &output_tensor));

    // Call the cuda kernel launcher
	ModCastFunctor(
        static_cast<int>(input_low_tensor.NumElements()),
        input_low_tensor.flat<float>().data(),
        input_high_tensor.flat<float>().data(),
		(double) q_, (double) p_,
        output_tensor->flat<float>().data());

  }

  private:
    float q_;
    float p_;
};

REGISTER_KERNEL_BUILDER(Name("ModCast").Device(DEVICE_GPU), ModCastOp);


REGISTER_OP("Fmod")
	.Attr("T: {float, double}")
	.Attr("p: float")
    .Input("inp: T")
    .Output("output: T")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(0));
      return Status::OK();
    });

template <typename T> void FmodFunctor(int size, const T* in, const T p, T* out);
template <> void FmodFunctor<float>(int size, const float* in, const float p, float* out);
template <> void FmodFunctor<double>(int size, const double* in, const double p, double* out);

template <typename T>
class FmodOp : public OpKernel {
 public:
  explicit FmodOp(OpKernelConstruction* context) : OpKernel(context) {
	OP_REQUIRES_OK(context, context->GetAttr("p", &p_));
  }

  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    const Tensor& input_tensor = context->input(0);

    // Create an output tensor
    Tensor* output_tensor = NULL;
	OP_REQUIRES_OK(context, context->forward_input_or_allocate_output(
                                {0}, 0, input_tensor.shape(), &output_tensor));

    // Call the cuda kernel launcher
	FmodFunctor<T>(
        static_cast<int>(input_tensor.NumElements()),
        input_tensor.flat<T>().data(),
		p_,
        output_tensor->flat<T>().data());

  }

  private:
    float p_;
};

REGISTER_KERNEL_BUILDER(Name("Fmod").Device(DEVICE_GPU).TypeConstraint<float>("T"), FmodOp<float>);
REGISTER_KERNEL_BUILDER(Name("Fmod").Device(DEVICE_GPU).TypeConstraint<double>("T"), FmodOp<double>);
