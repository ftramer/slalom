#define EIGEN_USE_THREADS

#include <iostream>
#include "tensorflow/core/framework/numeric_op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/common_runtime/threadpool_device.h"

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include <dlfcn.h>

using namespace std;
using namespace tensorflow;

template <typename Device, typename T>
class ReluSlalomOp : public OpKernel {
 public:
  explicit ReluSlalomOp(OpKernelConstruction* context) : OpKernel(context) {
#ifdef USE_SGX
    OP_REQUIRES_OK(context, context->GetAttr("eid_low", &eid_low_));
    OP_REQUIRES_OK(context, context->GetAttr("eid_high", &eid_high_));
	lib_ = dlopen("App/enclave_bridge.so", RTLD_NOW);
#else
	lib_ = dlopen("lib/sgxdnn.so", RTLD_NOW);
#endif
	OP_REQUIRES(context, lib_ != NULL, errors::Unknown("Unable to load .so"));
	OP_REQUIRES_OK(context, context->GetAttr("activation", &activation_));
  }
 
  void Compute(OpKernelContext* context) override {
    const Tensor& input = context->input(0);
    const Tensor& blind = context->input(1);

    //std::cout << "ReLU input: " << input.DebugString() << std::endl;
    //std::cout << "ReLU blind: " << blind.DebugString() << std::endl;

    Tensor* output = nullptr;

    if (activation_ == "avgpoolrelu" || activation_ == "avgpoolrelu6") {
    	auto output_shape = TensorShape({input.shape().dim_sizes()[0], 1, 1, input.shape().dim_sizes()[3]});
    	OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output));
    } else {
		OP_REQUIRES_OK(context, context->forward_input_or_allocate_output({0}, 0, input.shape(), &output));
    }
 
    const Device& d = context->eigen_device<Device>();

#ifdef USE_SGX
    unsigned long int eid_ = (eid_high_ << 32) + eid_low_;

	typedef void (*relu_ecall)(unsigned long int eid, float* in, float* out, float* blind, int num_elements, char* activation);
	dlerror();
	relu_ecall relu = (relu_ecall) dlsym(lib_, "slalom_relu");
	const char *dlsym_error = dlerror();
   	OP_REQUIRES(context, !dlsym_error, errors::Unknown("loading of relu failed: ", dlsym_error));	

	relu(eid_,
		 (float*) input.flat<T>().data(),
		 (float*) output->flat<T>().data(),
		 (float*) blind.flat<T>().data(),
		 input.NumElements(),
		 (char*) activation_.c_str());

#else

	typedef void (*relu_func)(float* in, float* out, float* blind, int num_elements, char* activation);
    dlerror();
    relu_func relu = (relu_func) dlsym(lib_, "slalom_relu");
    const char *dlsym_error = dlerror();
    OP_REQUIRES(context, !dlsym_error, errors::Unknown("loading of relu failed: ", dlsym_error));

    relu((float*) input.flat<T>().data(),
    	 (float*) output->flat<T>().data(),
		 (float*) blind.flat<T>().data(),
		 input.NumElements(),
		 (char*) activation_.c_str());

#endif
  }

 private:
  void* lib_;
  std::string activation_;

#ifdef USE_SGX
  int64 eid_low_;
  int64 eid_high_;
#endif
};

typedef Eigen::ThreadPoolDevice CPUDevice;

REGISTER_KERNEL_BUILDER(Name("ReluSlalom").Device(DEVICE_CPU), ReluSlalomOp<CPUDevice, float>);

REGISTER_OP("ReluSlalom")
#ifdef USE_SGX
    .Attr("eid_low: int")
    .Attr("eid_high: int")
#endif
	.Attr("activation: string")
    .Input("inp: float")
    .Input("blind: float")
    .Output("out: float")
    //.SetShapeFn(shape_inference::UnchangedShape);
	.SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
		string activation;
		Status s = c->GetAttr("activation", &activation);
		if (activation == "avgpoolrelu" || activation == "avgpoolrelu6") {
			::tensorflow::shape_inference::ShapeHandle input_shape;
			TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 4, &input_shape));
			::tensorflow::shape_inference::DimensionHandle batch = c->Dim(input_shape, 0);
			::tensorflow::shape_inference::DimensionHandle ch = c->Dim(input_shape, 3);
			c->set_output(0, c->MakeShape({batch, 1, 1, ch}));
		} else {
			c->set_output(0, c->input(0));
		}
		return Status::OK();
	});

#undef EIGEN_USE_THREADS
