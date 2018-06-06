#ifndef CUDA_FMOD_CAST_H_ 
#define CUDA_FMOD_CAST_H_ 

struct ModCastFunctor {
  void operator()(int size, const float* in_low, const float* in_high, const double q, const double p, float* out);
};

template <typename T>
struct FmodFunctor {
  void operator()(int size, const T* in, const T p, T* out);
};

#endif 
