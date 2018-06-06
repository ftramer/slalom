
#ifndef SGXDNN_SHAPES_H_
#define SGXDNN_SHAPES_H_

#include <unsupported/Eigen/CXX11/Tensor>

using namespace std;

void GetWindowedOutputSizeVerboseV2(int input_size, int filter_size,
                                    int dilation_rate, int stride,
									const Eigen::PaddingType& padding_type,
									int* output_size,
                                    int* padding_before,
                                    int* padding_after) {

  // See also the parallel implementation in GetWindowedOutputSizeFromDimsV2.
  int effective_filter_size = (filter_size - 1) * dilation_rate + 1;

  switch (padding_type) {
    case Eigen::PaddingType::PADDING_VALID:
      *output_size = (input_size - effective_filter_size + stride) / stride;
      *padding_before = *padding_after = 0;
      break;
    case Eigen::PaddingType::PADDING_SAME:
      *output_size = (input_size + stride - 1) / stride;
      const int padding_needed =
          std::max(0, (*output_size - 1) * stride + effective_filter_size - input_size);
      // For odd values of total padding, add more padding at the 'right'
      // side of the given dimension.
      *padding_before = padding_needed / 2;
      *padding_after = padding_needed - *padding_before;
      break;
  }
}

void GetWindowedOutputSizeVerbose(int input_size, int filter_size,
                                  int stride, const Eigen::PaddingType& padding_type,
                                  int* output_size, int* padding_before,
                                  int* padding_after) {
  return GetWindowedOutputSizeVerboseV2(input_size, filter_size,
                                        /*dilation_rate=*/1, stride,
                                        padding_type, output_size,
                                        padding_before, padding_after);
}

void GetWindowedOutputSize(int input_size, int filter_size, int stride,
						   const Eigen::PaddingType& padding_type,
						   int* output_size, int* padding_size) {
  int padding_after_unused;
  return GetWindowedOutputSizeVerbose(input_size, filter_size, stride,
                                      padding_type, output_size, padding_size,
                                      &padding_after_unused);
}

#endif
