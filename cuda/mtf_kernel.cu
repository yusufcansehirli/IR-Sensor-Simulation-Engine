// mtf_kernel.cu
// ─────────────────────────────────────────────────────────────────────────────
// Optik MTF uygulaması — separable Gaussian PSF konvolüsyon.
// Mimari belge §5.4 ile örtüşür.
//
// sigma_pixels = blur_spot_mrad × focal_length_mm / pixel_pitch_um
// ─────────────────────────────────────────────────────────────────────────────
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>

// ─── Shared memory boyutu ─────────────────────────────────────────────────────
constexpr int TILE_W     = 32;
constexpr int TILE_H     = 32;
constexpr int MAX_RADIUS = 8;    // Max blur yarıçapı (sigma ≤ 2.7 piksel)

// ─── X yönü Gaussian bulanıklık kerneli ──────────────────────────────────────
__global__ void gaussianBlurX(
    const float* __restrict__ in,
    float*       __restrict__ out,
    int rows, int cols,
    float sigma_pixels
) {
    extern __shared__ float s_row[];   // (TILE_W + 2*radius) float

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y;
    if (y >= rows) return;

    int radius = min((int)(3.0f * sigma_pixels + 0.5f), MAX_RADIUS);
    float inv_2s2 = 1.0f / (2.0f * sigma_pixels * sigma_pixels);

    // Paylaşımlı belleğe satırı yükle (halo dahil)
    int sx = threadIdx.x + radius;
    s_row[sx] = (x < cols) ? in[y * cols + x] : 0.0f;

    // Sol halo
    if (threadIdx.x < radius) {
        int nx = x - radius;
        s_row[threadIdx.x] = (nx >= 0) ? in[y * cols + nx] : 0.0f;
    }
    // Sağ halo
    if (threadIdx.x >= blockDim.x - radius) {
        int rx = x + radius;
        s_row[sx + radius] = (rx < cols) ? in[y * cols + rx] : 0.0f;
    }
    __syncthreads();

    if (x >= cols) return;

    // Gaussian ağırlıklı toplam
    float sum = 0.0f, weight = 0.0f;
    for (int dx = -radius; dx <= radius; dx++) {
        float w = expf(-dx * dx * inv_2s2);
        sum    += s_row[sx + dx] * w;
        weight += w;
    }
    out[y * cols + x] = sum / weight;
}

// ─── Y yönü Gaussian bulanıklık kerneli ──────────────────────────────────────
__global__ void gaussianBlurY(
    const float* __restrict__ in,
    float*       __restrict__ out,
    int rows, int cols,
    float sigma_pixels
) {
    extern __shared__ float s_col[];   // (TILE_H + 2*radius) float

    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int x = blockIdx.x;
    if (x >= cols) return;

    int radius = min((int)(3.0f * sigma_pixels + 0.5f), MAX_RADIUS);
    float inv_2s2 = 1.0f / (2.0f * sigma_pixels * sigma_pixels);

    int sy = threadIdx.y + radius;
    s_col[sy] = (y < rows) ? in[y * cols + x] : 0.0f;

    if (threadIdx.y < radius) {
        int ny = y - radius;
        s_col[threadIdx.y] = (ny >= 0) ? in[ny * cols + x] : 0.0f;
    }
    if (threadIdx.y >= blockDim.y - radius) {
        int ry = y + radius;
        s_col[sy + radius] = (ry < rows) ? in[ry * cols + x] : 0.0f;
    }
    __syncthreads();

    if (y >= rows) return;

    float sum = 0.0f, weight = 0.0f;
    for (int dy = -radius; dy <= radius; dy++) {
        float w = expf(-dy * dy * inv_2s2);
        sum    += s_col[sy + dy] * w;
        weight += w;
    }
    out[y * cols + x] = sum / weight;
}

// ─── Bloom kerneli (yoğun hedefler için halasyon) ─────────────────────────────
__global__ void bloomKernel(
    const float* __restrict__ in,
    float*       __restrict__ out,
    int rows, int cols,
    float bloom_threshold,
    float bloom_sigma
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= cols || y >= rows) return;

    int idx = y * cols + x;
    float val = in[idx];

    if (val > bloom_threshold) {
        // Bloom kaynağı: aşan kısım çevreye yayılır
        // (gerçek bloom ikinci bir gaussian pass gerektirir, burada lokal etki)
        float excess = val - bloom_threshold;
        float sigma  = bloom_sigma;
        int   radius = min((int)(2.0f * sigma + 0.5f), MAX_RADIUS);

        float spread = 0.0f;
        float total_w = 0.0f;
        for (int dy = -radius; dy <= radius; dy++) {
            for (int dx = -radius; dx <= radius; dx++) {
                int nx = min(max(x + dx, 0), cols - 1);
                int ny = min(max(y + dy, 0), rows - 1);
                float d2 = (float)(dx*dx + dy*dy);
                float w  = expf(-d2 / (2.0f * sigma * sigma));
                spread   += in[ny * cols + nx] * w;
                total_w  += w;
            }
        }
        out[idx] = val + 0.1f * excess * (spread / total_w);
    } else {
        out[idx] = val;
    }
}

// ─── HOST launcher ────────────────────────────────────────────────────────────
extern "C" void launchMTF(
    const float* d_in,
    float* d_tmp,
    float* d_out,
    int rows, int cols,
    float sigma_pixels,
    float bloom_threshold,
    float bloom_spread_pixels
) {
    int shm_x = (TILE_W + 2 * MAX_RADIUS) * sizeof(float);
    int shm_y = (TILE_H + 2 * MAX_RADIUS) * sizeof(float);

    // X geçişi
    dim3 bx(TILE_W, 1);
    dim3 gx((cols + TILE_W - 1) / TILE_W, rows);
    gaussianBlurX<<<gx, bx, shm_x>>>(d_in, d_tmp, rows, cols, sigma_pixels);

    // Y geçişi
    dim3 by(1, TILE_H);
    dim3 gy(cols, (rows + TILE_H - 1) / TILE_H);
    gaussianBlurY<<<gy, by, shm_y>>>(d_tmp, d_out, rows, cols, sigma_pixels);

    // Bloom
    if (bloom_threshold > 0.0f && bloom_spread_pixels > 0.0f) {
        float* d_bloom_tmp;
        cudaMalloc(&d_bloom_tmp, rows * cols * sizeof(float));
        dim3 bb(16, 16);
        dim3 gb((cols + 15) / 16, (rows + 15) / 16);
        bloomKernel<<<gb, bb>>>(d_out, d_bloom_tmp,
                                 rows, cols,
                                 bloom_threshold, bloom_spread_pixels);
        cudaMemcpy(d_out, d_bloom_tmp, rows * cols * sizeof(float),
                   cudaMemcpyDeviceToDevice);
        cudaFree(d_bloom_tmp);
    }
}
