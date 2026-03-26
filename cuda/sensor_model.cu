// sensor_model.cu
// ─────────────────────────────────────────────────────────────────────────────
// IR Sensör modeli — FPA yanıtı, NEdT temporal noise, FPN, AGC pipeline.
// Mimari belge §5.1, §5.2, §5.3 ile örtüşür.
// ─────────────────────────────────────────────────────────────────────────────
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <curand_kernel.h>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include "gpu_types.cuh"


// ─── curand state başlatma ────────────────────────────────────────────────────
__global__ void curandInitKernel(curandState* states,
                                  unsigned long long seed,
                                  int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    curand_init(seed, idx, 0, &states[idx]);
}

// ─── FPN haritası oluşturma ───────────────────────────────────────────────────
// Sabit bir tohum ile başlatılan Gaussian gürültü haritası
// Gerçek sensörde kalibrasyon verisinden gelir
__global__ void generateFPNKernel(float* d_fpn, curandState* states,
                                   int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    curandState localState = states[idx];
    d_fpn[idx] = curand_normal(&localState);   // N(0,1) — fpn_sigma ile çarpılır
    states[idx] = localState;
}

// ─── Ana sensör model kerneli ─────────────────────────────────────────────────
__global__ void sensorModelKernel(
    const float*   d_radiance,    // Giriş: W/m²/sr, (rows×cols)
    uint16_t*      d_output,      // Çıkış: dijital değer 0–65535
    SensorParamsGPU params,
    curandState*   d_rand_states,
    const float*   d_fpn_map,     // FPN haritası N(0,1) önceden hesaplanmış
    float          global_min,    // AGC min (histogram percentile)
    float          global_max,    // AGC max
    int rows, int cols
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= cols || y >= rows) return;

    int idx = y * cols + x;
    curandState localState = d_rand_states[idx];

    float L = d_radiance[idx];

    // ── [1] FPA Yanıtı: radyans → dijital sayı ───────────────────────────────
    // Basit lineer model:
    //   responsivity [DN/(W/m²/sr)] ∝ D* / (f/# × pixel_pitch)
    // D* [cm·Hz^0.5/W], pixel_pitch [μm → m], integration_time [ms → s]
    float responsivity = params.d_star             // cm·Hz^0.5/W
                       / (params.f_number          // f/#
                          * params.pixel_pitch_um * 1e-4f);  // μm → cm
    float signal = L * responsivity * params.integration_time_ms * 1e-3f;

    // ── [2] Temporal Noise (NEdT tabanlı) ────────────────────────────────────
    // NEdT = minimum tespit edilebilir sıcaklık farkı [mK]
    // σ_L ≈ (dL/dT) × NEdT / 1000
    // Yaklaşım: dL/dT ≈ 4L/T @ 300K (Wien sapma bölgesi)
    float dLdT = 4.0f * L / 300.0f;
    float sigma_temporal = dLdT * (params.nedt_mk * 1e-3f)
                         * responsivity * params.integration_time_ms * 1e-3f;
    float noise_temporal = curand_normal(&localState) * sigma_temporal;

    // ── [3] Fixed Pattern Noise ───────────────────────────────────────────────
    // FPN: piksel başı sabit offset (1/f spatial noise karakteri)
    float fpn = d_fpn_map[idx] * params.fpn_sigma_fraction * signal;

    float dn = signal + noise_temporal + fpn;

    // ── [4] AGC Normalizasyon ─────────────────────────────────────────────────
    // global_min / global_max AGC CPU tarafında histogram ile belirlenir
    float range = global_max - global_min;
    if (range < 1e-10f) range = 1.0f;
    dn = (dn - global_min) / range;

    // Gain / Level ayarı (display için)
    dn = (dn - params.agc_level) * params.agc_gain + 0.5f;

    // ── [5] Clamp → uint16 ───────────────────────────────────────────────────
    dn = fmaxf(0.0f, fminf(1.0f, dn));
    d_output[idx] = (uint16_t)(dn * 65535.0f);

    d_rand_states[idx] = localState;
}

// ─── uint16 → float normalizasyon ────────────────────────────────────────────
__global__ void normToFloatKernel(
    const uint16_t* d_in,
    float* d_out,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    d_out[idx] = (float)d_in[idx] / 65535.0f;
}

// ─── HOST: AGC min/max hesabı (basit percentile) ─────────────────────────────
// CPU tarafında histogram → global_min / global_max belirler
// (Gerçek GPU histogram daha sonra eklenebilir)
void computeAGCRange(const float* d_radiance_host_copy,
                     int n, float percentile_lo, float percentile_hi,
                     float& out_min, float& out_max) {
    // Basit: min/max kullan (histogram percentile CPU'da)
    out_min = 1e30f;
    out_max = -1e30f;
    for (int i = 0; i < n; i++) {
        float v = d_radiance_host_copy[i];
        if (v < out_min) out_min = v;
        if (v > out_max) out_max = v;
    }
    // Percentile yaklaşımı: %2 altını ve %98 üstünü kırp
    float range = out_max - out_min;
    out_min += range * (1.0f - percentile_hi);
    out_max -= range * (1.0f - percentile_hi);
}

// ─── HOST launchers ───────────────────────────────────────────────────────────
extern "C" {

void sensor_init_curand(curandState** d_states, unsigned long long seed, int n) {
    cudaMalloc(d_states, n * sizeof(curandState));
    int block = 256;
    int grid  = (n + block - 1) / block;
    curandInitKernel<<<grid, block>>>(*d_states, seed, n);
}

void sensor_generate_fpn(float** d_fpn, curandState* d_states, int n) {
    cudaMalloc(d_fpn, n * sizeof(float));
    // FPN için ayrı state (seed farklı)
    curandState* d_fpn_states;
    cudaMalloc(&d_fpn_states, n * sizeof(curandState));
    int block = 256;
    int grid  = (n + block - 1) / block;
    curandInitKernel<<<grid, block>>>(d_fpn_states, 0xDEADBEEF, n);
    generateFPNKernel<<<grid, block>>>(*d_fpn, d_fpn_states, n);
    cudaFree(d_fpn_states);
}

void sensor_run(
    const float* d_radiance,
    uint16_t* d_output,
    curandState* d_rand_states,
    const float* d_fpn_map,
    SensorParamsGPU params,
    float global_min, float global_max,
    int rows, int cols
) {
    dim3 block(16, 16);
    dim3 grid((cols + 15) / 16, (rows + 15) / 16);
    sensorModelKernel<<<grid, block>>>(
        d_radiance, d_output, params,
        d_rand_states, d_fpn_map,
        global_min, global_max,
        rows, cols
    );
}

void sensor_uint16_to_float(const uint16_t* d_in, float* d_out, int n) {
    int block = 256;
    int grid  = (n + block - 1) / block;
    normToFloatKernel<<<grid, block>>>(d_in, d_out, n);
}

} // extern "C"
