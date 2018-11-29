#ifndef SGXDNN_MEMPOOL_H_
#define SGXDNN_MEMPOOL_H_

#define EIGEN_USE_TENSOR
#include <unsupported/Eigen/CXX11/Tensor>

namespace SGXDNN
{

	class MemPool
	{
	public:
		explicit MemPool(size_t max_chunks, size_t max_chunk_size) {
			allocated_bytes = 0;
		}

		virtual ~MemPool(){

		}

		template <typename T>
		T* alloc(size_t size) {
			T* ptr = (T*) Eigen::internal::aligned_malloc(size * sizeof(T));
			allocated_bytes += size * sizeof(T);
			return ptr;
		}

		template <typename T>
		T* realloc(T *ptr, size_t new_size, size_t old_size) {
			void* new_ptr = Eigen::internal::aligned_realloc((void*) ptr, new_size, old_size);
			return (T*) new_ptr;
		}

		template <typename T>
		void release(T* mem_loc) {
			Eigen::internal::aligned_free(mem_loc);
		}

		long allocated_bytes;

	};
}
#endif
