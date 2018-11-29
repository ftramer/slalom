#include "aes-stream.hpp"
#include "sgxdnn_main.hpp"

#if defined(__GNUC__) && !defined(__clang__)
# pragma GCC target("ssse3")
# pragma GCC target("aes")
#endif

#include <immintrin.h>
#include <stdint.h>
#include <stdlib.h>
#include "assert.h"
#include <cstdio>
#include <string>

#ifdef USE_SGX
#include "Enclave.h"
#endif

#define COMPILER_ASSERT(X) (void) sizeof(char[(X) ? 1 : -1])

#if defined(__IBMC__) || defined(__SUNPRO_C) || defined(__SUNPRO_CC)
# pragma pack(1)
#else
# pragma pack(push, 1)
#endif

typedef struct CRYPTO_ALIGN(16) _aes_stream_state {
    __m128i round_keys[AES_STREAM_ROUNDS + 1];
    __m128i counter;
} _aes_stream_state;

#if defined(__IBMC__) || defined(__SUNPRO_C) || defined(__SUNPRO_CC)
# pragma pack()
#else
# pragma pack(pop)
#endif

#define DRC(ROUND, RC)                                       \
    do {                                                     \
        s = _mm_aeskeygenassist_si128(t1, (RC));             \
        round_keys[ROUND] = t1;                              \
        t1 = _mm_xor_si128(t1, _mm_slli_si128(t1, 4));       \
        t1 = _mm_xor_si128(t1, _mm_slli_si128(t1, 8));       \
        t1 = _mm_xor_si128(t1, _mm_shuffle_epi32(s, 0xff));  \
    } while (0)

#define DRC2(ROUND, RC)                                      \
    do {                                                     \
        s = _mm_aeskeygenassist_si128(t2, (RC));             \
        round_keys[ROUND] = t2;                              \
        t2 = _mm_xor_si128(t2, _mm_slli_si128(t1, 4));       \
        t2 = _mm_xor_si128(t2, _mm_slli_si128(t1, 8));       \
        t2 = _mm_xor_si128(t2, _mm_shuffle_epi32(s, 0xaa));  \
    } while (0)

#if AES_STREAM_ROUNDS == 10
static void
_aes_key_expand_128(__m128i round_keys[AES_STREAM_ROUNDS + 1], __m128i t1)
{
    __m128i s;

    DRC(0, 1); DRC(1, 2); DRC(2, 4); DRC(3, 8); DRC(4, 16);
    DRC(5, 32); DRC(6, 64); DRC(7, 128); DRC(8, 27); DRC(9, 54);
    round_keys[10] = t1;
}

#elif AES_STREAM_ROUNDS == 14

static void
_aes_key_expand_256(__m128i round_keys[AES_STREAM_ROUNDS + 1],
                    __m128i t1, __m128i t2)
{
    __m128i s;

    round_keys[0] = t1;
    DRC(1, 1); DRC2(2, 1); DRC(3, 2); DRC2(4, 2);
    DRC(5, 4); DRC2(6, 4); DRC(7, 8); DRC2(8, 8);
    DRC(9, 16); DRC2(10, 16); DRC(11, 32); DRC2(12, 32);
    DRC(13, 64);
    round_keys[14] = t1;
}
#endif

static void
_aes_stream(_aes_stream_state *_st, unsigned char *buf, size_t buf_len, bool shift)
{
    CRYPTO_ALIGN(16) unsigned char t[16];
    const __m128i  one = _mm_set_epi64x(0, 1);
    const __m128i  two = _mm_set_epi64x(0, 2);
    __m128i       *round_keys = _st->round_keys;
    __m128i        c0, c1, c2, c3, c4, c5, c6, c7;
    __m128i        r0, r1, r2, r3, r4, r5, r6, r7;
    __m128i        s0, s1, s2, s3, s4, s5, s6, s7;
    size_t         i;
    size_t         remaining;

#if AES_STREAM_ROUNDS == 10
# define COMPUTE_AES_STREAM_ROUNDS(N)                                                  \
    do {                                                                               \
        r##N = _mm_aesenc_si128(   _mm_xor_si128(c##N, round_keys[0]), round_keys[1]); \
        r##N = _mm_aesenc_si128(_mm_aesenc_si128(r##N, round_keys[2]), round_keys[3]); \
        r##N = _mm_aesenc_si128(_mm_aesenc_si128(r##N, round_keys[4]), round_keys[5]); \
        s##N = r##N;                                                                   \
        r##N = _mm_aesenc_si128(_mm_aesenc_si128(r##N, round_keys[6]), round_keys[7]); \
        r##N = _mm_aesenc_si128(_mm_aesenc_si128(r##N, round_keys[8]), round_keys[9]); \
        r##N = _mm_xor_si128(s##N, _mm_aesenclast_si128(r##N, round_keys[10]));        \
    } while (0)

#elif AES_STREAM_ROUNDS == 14

# define COMPUTE_AES_STREAM_ROUNDS(N)                                                    \
    do {                                                                                 \
        r##N = _mm_aesenc_si128(   _mm_xor_si128(c##N, round_keys[ 0]), round_keys[ 1]); \
        r##N = _mm_aesenc_si128(_mm_aesenc_si128(r##N, round_keys[ 2]), round_keys[ 3]); \
        r##N = _mm_aesenc_si128(_mm_aesenc_si128(r##N, round_keys[ 4]), round_keys[ 5]); \
        r##N = _mm_aesenc_si128(_mm_aesenc_si128(r##N, round_keys[ 6]), round_keys[ 7]); \
        s##N = r##N;                                                                     \
        r##N = _mm_aesenc_si128(_mm_aesenc_si128(r##N, round_keys[ 8]), round_keys[ 9]); \
        r##N = _mm_aesenc_si128(_mm_aesenc_si128(r##N, round_keys[10]), round_keys[11]); \
        r##N = _mm_aesenc_si128(_mm_aesenc_si128(r##N, round_keys[12]), round_keys[13]); \
        r##N = _mm_xor_si128(s##N, _mm_aesenclast_si128(r##N, round_keys[14]));          \
    } while (0)
#endif

    c0 = _st->counter;
    remaining = buf_len;
    while (remaining >= 128) {
        c1 = _mm_add_epi64(c0, one);
        c2 = _mm_add_epi64(c0, two);
        c3 = _mm_add_epi64(c2, one);
        c4 = _mm_add_epi64(c2, two);
        c5 = _mm_add_epi64(c4, one);
        c6 = _mm_add_epi64(c4, two);
        c7 = _mm_add_epi64(c6, one);
        COMPUTE_AES_STREAM_ROUNDS(0);
        COMPUTE_AES_STREAM_ROUNDS(1);
        COMPUTE_AES_STREAM_ROUNDS(2);
        COMPUTE_AES_STREAM_ROUNDS(3);
        COMPUTE_AES_STREAM_ROUNDS(4);
        COMPUTE_AES_STREAM_ROUNDS(5);
        COMPUTE_AES_STREAM_ROUNDS(6);
        COMPUTE_AES_STREAM_ROUNDS(7);
        c0 = _mm_add_epi64(c7, one);

		if (shift) {
			r0 = _mm_srai_epi32(r0, 9);
			r1 = _mm_srai_epi32(r1, 9);
			r2 = _mm_srai_epi32(r2, 9);
			r3 = _mm_srai_epi32(r3, 9);
			r4 = _mm_srai_epi32(r4, 9);
			r5 = _mm_srai_epi32(r5, 9);
			r6 = _mm_srai_epi32(r6, 9);
			r7 = _mm_srai_epi32(r7, 9);
			_mm_store_ps((float *) (buf +   0), _mm_cvtepi32_ps(r0));
        	_mm_store_ps((float *) (buf +  16), _mm_cvtepi32_ps(r1));
        	_mm_store_ps((float *) (buf +  32), _mm_cvtepi32_ps(r2));
        	_mm_store_ps((float *) (buf +  48), _mm_cvtepi32_ps(r3));
        	_mm_store_ps((float *) (buf +  64), _mm_cvtepi32_ps(r4));
        	_mm_store_ps((float *) (buf +  80), _mm_cvtepi32_ps(r5));
        	_mm_store_ps((float *) (buf +  96), _mm_cvtepi32_ps(r6));
        	_mm_store_ps((float *) (buf + 112), _mm_cvtepi32_ps(r7));
		} else {
        	_mm_storeu_si128((__m128i *) (void *) (buf +   0), r0);
        	_mm_storeu_si128((__m128i *) (void *) (buf +  16), r1);
        	_mm_storeu_si128((__m128i *) (void *) (buf +  32), r2);
        	_mm_storeu_si128((__m128i *) (void *) (buf +  48), r3);
        	_mm_storeu_si128((__m128i *) (void *) (buf +  64), r4);
        	_mm_storeu_si128((__m128i *) (void *) (buf +  80), r5);
        	_mm_storeu_si128((__m128i *) (void *) (buf +  96), r6);
        	_mm_storeu_si128((__m128i *) (void *) (buf + 112), r7);
		}
        buf += 128;
        remaining -= 128;
    }
    while (remaining >= 32) {
        c1 = _mm_add_epi64(c0, one);
        COMPUTE_AES_STREAM_ROUNDS(0);
        COMPUTE_AES_STREAM_ROUNDS(1);
        c0 = _mm_add_epi64(c1, one);
	
		if (shift) {
            r0 = _mm_srai_epi32(r0, 9);
            r1 = _mm_srai_epi32(r1, 9);
			_mm_store_ps((float *) (buf +   0), _mm_cvtepi32_ps(r0));
        	_mm_store_ps((float *) (buf +  16), _mm_cvtepi32_ps(r1));
		} else {
        	_mm_storeu_si128((__m128i *) (void *) (buf +  0), r0);
        	_mm_storeu_si128((__m128i *) (void *) (buf + 16), r1);
		}
        buf += 32;
        remaining -= 32;
    }
    while (remaining >= 16) {
        COMPUTE_AES_STREAM_ROUNDS(0);
        c0 = _mm_add_epi64(c0, one);

		if (shift) {
            r0 = _mm_srai_epi32(r0, 9);
			_mm_store_ps((float *) (buf +   0), _mm_cvtepi32_ps(r0));
		} else {
        	_mm_storeu_si128((__m128i *) (void *) buf, r0);
		}
        buf += 16;
        remaining -= 16;
    }
    if (remaining > (size_t) 0U) {
        COMPUTE_AES_STREAM_ROUNDS(0);
        c0 = _mm_add_epi64(c0, one);

        _mm_store_si128((__m128i *) (void *) t, r0);

		assert(!shift);
        for (i = 0; i < remaining; i++) {
          	buf[i] = t[i];
        }
    }
    _st->counter = c0;

    c0 = _mm_xor_si128(c0, _mm_set_epi64x(1ULL << 63, 0));

#if AES_STREAM_ROUNDS == 10
    COMPUTE_AES_STREAM_ROUNDS(0);
    _aes_key_expand_128(round_keys, r0);

#elif AES_STREAM_ROUNDS == 14

    c1 = _mm_add_epi64(c0, one);
    COMPUTE_AES_STREAM_ROUNDS(0);
    COMPUTE_AES_STREAM_ROUNDS(1);
    _aes_key_expand_256(round_keys, r0, r1);
#endif
}

void
aes_stream_init(aes_stream_state *st,
                const unsigned char seed[AES_STREAM_SEEDBYTES])
{
    _aes_stream_state *_st = (_aes_stream_state *) (void *) st;

    COMPILER_ASSERT(sizeof *st >= sizeof *_st);

#if AES_STREAM_ROUNDS == 10
    _aes_key_expand_128(_st->round_keys,
                        _mm_loadu_si128((const __m128i *) (const void *) seed));
    _st->counter = _mm_loadu_si128((const __m128i *) (const void *) (seed + 16));

#elif AES_STREAM_ROUNDS == 14

    _aes_key_expand_256(_st->round_keys,
                        _mm_loadu_si128((const __m128i *) (const void *) seed),
                        _mm_loadu_si128((const __m128i *) (const void *) (seed + 16)));
    _st->counter = _mm_setzero_si128();
#endif
}

/*
 * Unblind one float vector
 */
inline __m256 unblind(__m256 inp8f, __m256 blind8f) {
	__m256 sub8f = _mm256_sub_ps(inp8f, blind8f);         // unblinded
	__m256 if_geq = _mm256_cmp_ps(sub8f, mid8f, 0x0d);    // unblinded >= mid
	__m256 if_lt = _mm256_cmp_ps(sub8f, negmid8f, 0x01);  // unblinded < -mid
	__m256 then8f = _mm256_sub_ps(sub8f, p8f);            // unblinded - p
	__m256 elif8f = _mm256_add_ps(sub8f, p8f);            // unblinded + p
	__m256 res8f = _mm256_blendv_ps(
								_mm256_blendv_ps(
										sub8f,
										elif8f,
										if_lt),
								then8f,
								if_geq);

	return res8f;
}

/*
 * Re-blind one float vector
 */
inline void reblind(__m256 act_res8f, __m128 rand1, __m128 rand2, float* out) {
	__m256 new_rand8f = _mm256_castps128_ps256(rand1);
	new_rand8f = _mm256_insertf128_ps(new_rand8f, rand2, 1);	

	__m256 out_blind8f = _mm256_add_ps(act_res8f, new_rand8f);
	__m256 if_geq = _mm256_cmp_ps(out_blind8f, mid8f, 0x0d);
	__m256 then8f = _mm256_sub_ps(out_blind8f, p8f);            // blinded - p

	__m256 res8f = _mm256_blendv_ps(out_blind8f, then8f, if_geq);
	_mm256_stream_ps(out, res8f);
}

# define COMPUTE_AES_STREAM_ROUNDS(N)                                              \
do {                                                                               \
	r##N = _mm_aesenc_si128(   _mm_xor_si128(c##N, round_keys[0]), round_keys[1]); \
	r##N = _mm_aesenc_si128(_mm_aesenc_si128(r##N, round_keys[2]), round_keys[3]); \
	r##N = _mm_aesenc_si128(_mm_aesenc_si128(r##N, round_keys[4]), round_keys[5]); \
	s##N = r##N;                                                                   \
	r##N = _mm_aesenc_si128(_mm_aesenc_si128(r##N, round_keys[6]), round_keys[7]); \
	r##N = _mm_aesenc_si128(_mm_aesenc_si128(r##N, round_keys[8]), round_keys[9]); \
	r##N = _mm_xor_si128(s##N, _mm_aesenclast_si128(r##N, round_keys[10]));        \
} while (0)

#define AES_STREAM_FUSED(_st, out, blinded_input, blind, image_size, ch, 	\
						 act_func, func_x, func_z, func_z_outer, params)	\
	assert(AES_STREAM_ROUNDS == 10);					\
	assert(ch % 32 == 0);								\
\
    CRYPTO_ALIGN(16) unsigned char t[16];				\
    const __m128i  one = _mm_set_epi64x(0, 1);			\
    const __m128i  two = _mm_set_epi64x(0, 2);			\
    __m128i       *round_keys = _st->round_keys;		\
    __m128i        c0, c1, c2, c3, c4, c5, c6, c7;		\
    __m128i        r0, r1, r2, r3, r4, r5, r6, r7;		\
    __m128i        s0, s1, s2, s3, s4, s5, s6, s7;		\
    __m256		   m0, m1, m2, m3;						\
    size_t         i;									\
    size_t         remaining;							\
	size_t buf_len = image_size * ch * sizeof(float);	\
    c0 = _st->counter;									\
    remaining = buf_len;								\
\
	for(int i=0; i<image_size; i++) {		\
		for(int j=0; j<ch; j+=32) {			\
			c1 = _mm_add_epi64(c0, one);	\
			c2 = _mm_add_epi64(c0, two);	\
			c3 = _mm_add_epi64(c2, one);	\
			c4 = _mm_add_epi64(c2, two);	\
			c5 = _mm_add_epi64(c4, one);	\
			c6 = _mm_add_epi64(c4, two);	\
			c7 = _mm_add_epi64(c6, one);	\
			COMPUTE_AES_STREAM_ROUNDS(0);	\
			COMPUTE_AES_STREAM_ROUNDS(1);	\
			COMPUTE_AES_STREAM_ROUNDS(2);	\
			COMPUTE_AES_STREAM_ROUNDS(3);	\
			COMPUTE_AES_STREAM_ROUNDS(4);	\
			COMPUTE_AES_STREAM_ROUNDS(5);	\
			COMPUTE_AES_STREAM_ROUNDS(6);	\
			COMPUTE_AES_STREAM_ROUNDS(7);	\
			c0 = _mm_add_epi64(c7, one);	\
\
			r0 = _mm_srai_epi32(r0, 9);	\
			r1 = _mm_srai_epi32(r1, 9);	\
			r2 = _mm_srai_epi32(r2, 9);	\
			r3 = _mm_srai_epi32(r3, 9);	\
			r4 = _mm_srai_epi32(r4, 9);	\
			r5 = _mm_srai_epi32(r5, 9);	\
			r6 = _mm_srai_epi32(r6, 9);	\
			r7 = _mm_srai_epi32(r7, 9);	\
\
			m0 = unblind(_mm256_load_ps(blinded_input +  0), _mm256_load_ps(blind +  0));	\
			m1 = unblind(_mm256_load_ps(blinded_input +  8), _mm256_load_ps(blind +  8));	\
			m2 = unblind(_mm256_load_ps(blinded_input + 16), _mm256_load_ps(blind + 16));	\
			m3 = unblind(_mm256_load_ps(blinded_input + 24), _mm256_load_ps(blind + 24));	\
\
			func_z(params.res_z, params.temp_z, params.r_right_data, m0, i, j	  , ch);	\
			func_z(params.res_z, params.temp_z, params.r_right_data, m1, i, j+8 , ch);	\
			func_z(params.res_z, params.temp_z, params.r_right_data, m2, i, j+16, ch);	\
			func_z(params.res_z, params.temp_z, params.r_right_data, m3, i, j+24, ch);	\
\
			m0 = act_func(m0);	\
			m1 = act_func(m1);	\
			m2 = act_func(m2);	\
			m3 = act_func(m3);	\
\
			func_x(params.res_x, params.temp_x, params.kernel_r_data, m0, i, j   , image_size, ch);	\
			func_x(params.res_x, params.temp_x, params.kernel_r_data, m1, i, j+8 , image_size, ch);	\
			func_x(params.res_x, params.temp_x, params.kernel_r_data, m2, i, j+16, image_size, ch);	\
			func_x(params.res_x, params.temp_x, params.kernel_r_data, m3, i, j+24, image_size, ch);	\
\
			reblind(m0, _mm_cvtepi32_ps(r0), _mm_cvtepi32_ps(r1), out +  0);	\
			reblind(m1, _mm_cvtepi32_ps(r2), _mm_cvtepi32_ps(r3), out +  8);	\
			reblind(m2, _mm_cvtepi32_ps(r4), _mm_cvtepi32_ps(r5), out + 16);	\
			reblind(m3, _mm_cvtepi32_ps(r6), _mm_cvtepi32_ps(r7), out + 24);	\
\
			out += 32;				\
			blinded_input += 32;	\
			blind += 32;			\
		}																					\
		func_z_outer(params.res_z, params.temp_z, params.r_left_data, i, image_size);	\
	}																						\
    _st->counter = c0;										\
    c0 = _mm_xor_si128(c0, _mm_set_epi64x(1ULL << 63, 0));	\
    COMPUTE_AES_STREAM_ROUNDS(0);							\
    _aes_key_expand_128(round_keys, r0);					\

void
aes_stream(aes_stream_state *st, unsigned char *buf, size_t buf_len, bool shift)
{
    _aes_stream((_aes_stream_state *) (void *) st, buf, buf_len, shift);
}

/* reblinds the input and writes it to output outside of the enclave. For efficiency reasons, we perform a single
 * loop over the data in which we:
 * 	- compute activations
 *  - compute the AES PRG stream and blind the activations
 *  - perform the Freivalds checks for integrity
 *  - write the blinded data outside of the enclave
 */
void aes_stream_fused(aes_stream_state *st, float* out, float* input, float* blind, size_t image_size, size_t ch,
					  char* activation, integrityParams& params) {
	std::string act(activation);
	if (!params.integrity) {
		if (act == "relu") {
			AES_STREAM_FUSED(((_aes_stream_state *) (void *) st), out, input, blind, image_size, ch,
							 relu_avx, empty_verif_x, empty_verif_z, empty_verif_z_outer, params);
		} else {
			AES_STREAM_FUSED(((_aes_stream_state *) (void *) st), out, input, blind, image_size, ch,
							 relu6_avx, empty_verif_x, empty_verif_z, empty_verif_z_outer, params);
		}
		return;
	}

	assert(act == "relu");
	if (params.pointwise_x && params.pointwise_z) {
		AES_STREAM_FUSED(((_aes_stream_state *) (void *) st), out, input, blind, image_size, ch,
						 relu_avx, preproc_verif_pointwise_X_inner, preproc_verif_pointwise_Z_inner, empty_verif_z_outer, params);
	} else if (params.pointwise_x) {
		AES_STREAM_FUSED(((_aes_stream_state *) (void *) st), out, input, blind, image_size, ch,
						 relu_avx, preproc_verif_pointwise_X_inner, preproc_verif_Z_inner, preproc_verif_Z_outer, params);
	} else if (params.pointwise_z) {
		AES_STREAM_FUSED(((_aes_stream_state *) (void *) st), out, input, blind, image_size, ch,
						 relu_avx, preproc_verif_X_inner, preproc_verif_pointwise_Z_inner, empty_verif_z_outer, params);
	} else {
		AES_STREAM_FUSED(((_aes_stream_state *) (void *) st), out, input, blind, image_size, ch,
						 relu_avx, preproc_verif_X_inner, preproc_verif_Z_inner, preproc_verif_Z_outer, params);
	}
}
