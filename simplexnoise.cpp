#include <stdint.h>
#include <array>

#define SIMDPP_ARCH_X86_SSE3
#include "simdpp/simd.h"


using vec4 = simdpp::float32<4>;
using vec16 = simdpp::float32<16>;
using ivec4 = simdpp::int32<4>;
using ivec16 = simdpp::int32<16>;

// Generates hashes for a vector of integers
static ivec16 hashblock(const ivec16& a)
{
    simdpp::uint32<16> h = simdpp::to_uint32(a);

    h = (h + 0x7ed55d16) + (h << 12);
    h = (h ^ 0xc761c23c) ^ (h >> 19);
    h = (h + 0x165667b1) + (h << 5);
    h = (h + 0xd3a2646c) ^ (h << 9);
    h = (h + 0xfd7046c5) + (h << 3);
    h = (h ^ 0xb55a4f09) ^ (h >> 16);
    return simdpp::to_int32(h);
}

// Generates gradients for a vector of hashes
static vec16 grad_block(const ivec16& hashes, const vec16& x, const vec16& y)
{
    vec16 mult_1 = (simdpp::to_float32(hashes & 1) - 0.5);
    vec16 mult_2 = (simdpp::to_float32(hashes & 2) - 1.0);
    simdpp::mask_int32<16> swap_mask = (hashes & 4) > 0;

    vec16 mult_x = simdpp::blend(mult_1, mult_2, swap_mask);
    vec16 mult_y = simdpp::blend(mult_2, mult_1, swap_mask);

    return x * mult_x + y * mult_y;
}

static vec16 pow2_vec16(const vec16& v)
{
    return v * v;
}

static vec16 pow4_vec16(const vec16& v)
{
    return pow2_vec16(pow2_vec16(v));
}

// Skewing/Unskewing factors for 2D
constexpr float F2 = 0.366025403f;  // F2 = (sqrt(3) - 1) / 2
constexpr float G2 = 0.211324865f;  // G2 = (3 - sqrt(3)) / 6   = F2 / (1 + 2 * K)

const vec4 v_4step = simdpp::make_float(0.0, 0.25, 0.5, 0.75);
const ivec16 ivec16_0 = simdpp::splat(0);
const ivec16 ivec16_1 = simdpp::splat(1);
const vec16 vec16_half = simdpp::splat(0.5);
const vec16 vec16_0 = simdpp::splat(0.0f);

const vec16 vec16_xscale = simdpp::make_float(
    0.0, 0.25, 0.5, 0.75,
    0.0, 0.25, 0.5, 0.75,
    0.0, 0.25, 0.5, 0.75,
    0.0, 0.25, 0.5, 0.75
);
const vec16 vec16_yscale = simdpp::make_float(
    0.0,  0.0,  0.0,  0.0,
    0.25, 0.25, 0.25, 0.25,
    0.5,  0.5,  0.5,  0.5,
    0.75, 0.75, 0.75, 0.75
);

/*
 * Generates a 4x4 block of simplex noise.
 * End point is not actually calculated, basically just think of it like an
 * integer range.
 */
std::array<float, 16> noiseblock(float x_begin, float y_begin, float x_end, float y_end) {
    float x_diff = x_end - x_begin;
    float y_diff = y_end - y_begin;

    // Calculate x and y values for 4x4 block
    vec16 x = vec16_xscale * x_diff + x_begin;
    vec16 y = vec16_yscale * y_diff + y_begin;

    // Skew the input space to determine which simplex cell we're in
    vec16 s = (x + y) * F2;  // Hairy factor for 2D
    vec16 xs = x + s;
    vec16 ys = y + s;
    vec16 i_fl = simdpp::floor(xs);
    vec16 j_fl = simdpp::floor(ys);
    ivec16 i = simdpp::to_int32(i_fl);
    ivec16 j = simdpp::to_int32(j_fl);

    // Unskew the cell origin back to (x,y) space
    vec16 t = (i_fl + j_fl) * G2;
    vec16 X0 = simdpp::to_float32(i) - t;
    vec16 Y0 = simdpp::to_float32(j) - t;
    vec16 x0 = x - X0;  // The x,y distances from the cell origin
    vec16 y0 = y - Y0;

    // For the 2D case, the simplex shape is an equilateral triangle.
    // Determine which simplex we are in.
    // Offsets for second (middle) corner of simplex in (i,j) coords
    simdpp::mask_float32<16> x0y0_mask = x0 > y0;
    ivec16 i1 = simdpp::blend(ivec16_1, ivec16_0, x0y0_mask);
    ivec16 j1 = simdpp::blend(ivec16_0, ivec16_1, x0y0_mask);

    // A step of (1,0) in (i,j) means a step of (1-c,-c) in (x,y), and
    // a step of (0,1) in (i,j) means a step of (-c,1-c) in (x,y), where
    // c = (3-sqrt(3))/6

    vec16 x1 = x0 - simdpp::to_float32(i1) + G2;   // Offsets for middle corner in (x,y) unskewed coords
    vec16 y1 = y0 - simdpp::to_float32(j1) + G2;
    vec16 x2 = x0 - 1.0f + 2.0f * G2;   // Offsets for last corner in (x,y) unskewed coords
    vec16 y2 = y0 - 1.0f + 2.0f * G2;

    ivec16 gi0 = hashblock(i + hashblock(j));
    ivec16 gi1 = hashblock(i + i1 + hashblock(j + j1));
    ivec16 gi2 = hashblock(i + 1 + hashblock(j + 1));

    // Calculate the contribution from the first corner
    vec16 t0 = vec16_half - pow2_vec16(x0) - pow2_vec16(y0);
    vec16 n0_pre = pow4_vec16(t0) * grad_block(gi0, x0, y0); // Values of n0 if t0 >= 0
    vec16 n0 = simdpp::blend(vec16_0, n0_pre, t0 < 0.0f);

    // Calculate the contribution from the second corner
    vec16 t1 = vec16_half - pow2_vec16(x1) - pow2_vec16(y1);
    vec16 n1_pre = pow4_vec16(t1) * grad_block(gi1, x1, y1);
    vec16 n1 = simdpp::blend(vec16_0, n1_pre, t1 < 0.0f);

    // Calculate the contribution from the third corner
    vec16 t2 = vec16_half - pow2_vec16(x2) - pow2_vec16(y2);
    vec16 n2_pre = pow4_vec16(t2) * grad_block(gi2, x2, y2);
    vec16 n2 = simdpp::blend(vec16_0, n2_pre, t2 < 0.0f);

    // Add contributions from each corner to get the final noise value.
    // The result is scaled to return values in the interval [-1,1].
    vec16 n_sum = (n0 + n1 + n2) * 90.4613f;

    SIMDPP_ALIGN(16);
    std::array<float, 16> block_out;
    simdpp::store(block_out.data(), n_sum);
    return block_out;
}
