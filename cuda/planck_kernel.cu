// planck_kernel.cu
// ─────────────────────────────────────────────────────────────────────────────
// Planck kara cisim radyansı ve bant-entegre radyans hesabı.
// Mimari belge §2.1, §2.2, §2.3 ile örtüşür.
// ─────────────────────────────────────────────────────────────────────────────
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>
#include <cstdint>
#include "gpu_types.cuh"

// ─── Fizik sabitleri ─────────────────────────────────────────────────────────
__constant__ constexpr float kH  = 6.62607015e-34f;  // Planck sabiti [J·s]
__constant__ constexpr float kC  = 2.99792458e+8f;   // Işık hızı [m/s]
__constant__ constexpr float kKb = 1.380649e-23f;    // Boltzmann sabiti [J/K]
__constant__ constexpr float kSb = 5.670374419e-8f;  // Stefan-Boltzmann [W/m²/K⁴]
__constant__ constexpr float kPI = 3.14159265358979f;
__constant__ constexpr float kSolarIrr = 1361.0f;    // AM0 güneş konstantı [W/m²]

// ─── Tek noktada Planck kara cisim emisyonu ──────────────────────────────────
// lambda_m: dalga boyu [metre]
// Dönüş: [W/m²/sr/m] (spektral radyans)
__device__ __forceinline__ float planck(float T, float lambda_m) {
    // Wien'in taşma noktasını kontrol et
    if (T <= 0.0f || lambda_m <= 0.0f) return 0.0f;
    float c1 = 2.0f * kH * kC * kC;
    float c2 = (kH * kC) / (kKb * T);
    float exponent = c2 / lambda_m;
    // exp taşmasını önle
    if (exponent > 87.0f) return 0.0f;
    return (c1 / (powf(lambda_m, 5.0f))) / (expf(exponent) - 1.0f);
}

// ─── Simpson kuralıyla bant entegrasyonu ─────────────────────────────────────
// N noktanın tek sayı olması gerekir, minimum 5
// Dönüş: [W/m²/sr] (bant radyansı)
__device__ float bandIntegrate(float T, float eps,
                                float lam_min_um, float lam_max_um,
                                int N = 7) {
    // lambda μm → metre
    float lam_min = lam_min_um * 1e-6f;
    float lam_max = lam_max_um * 1e-6f;
    float dlam = (lam_max - lam_min) / (N - 1);

    float sum = 0.0f;
    for (int i = 0; i < N; i++) {
        float lam = lam_min + i * dlam;
        float L   = eps * planck(T, lam);
        // Simpson katsayıları: 1, 4, 2, 4, 2, ..., 4, 1
        float coeff = (i == 0 || i == N - 1) ? 1.0f :
                      (i % 2 == 1)           ? 4.0f : 2.0f;
        sum += coeff * L;
    }
    return sum * dlam / 3.0f;
}

// ─── SWIR solar yansıması ─────────────────────────────────────────────────────
// SWIR'de emisyon düşük, güneş yansıması dominant.
// Basit Lambertian yansıma: L_ref = (1 - eps) * E_sol * cos(theta) / pi
__device__ float swirReflectedRadiance(float eps,
                                        float E_solar,      // W/m²
                                        float cos_sun) {
    return (1.0f - eps) * E_solar * fmaxf(cos_sun, 0.0f) / kPI;
}

// ─────────────────────────────────────────────────────────────────────────────
// GPU KERNEL: Her piksel için bant-entegre radyans hesapla
// ─────────────────────────────────────────────────────────────────────────────

// Constant memory — materyal tablosu (gpu_types.cuh ile aynı yapı)
__constant__ GPUMaterialEntry d_materials[256];  // Max 256 materyal

__global__ void planckRadianceKernel(
    const float*    d_T_surface,      // Yüzey sıcaklıkları [K], (rows×cols)
    const int*      d_mat_id,         // Materyal ID haritası (rows×cols)
    float*          d_radiance,       // Çıkış: bant radyansı [W/m²/sr]
    BandParams      band,
    float           T_ambient_K,      // Atmosfer sıcaklığı
    float           tau_atm,          // Atmosferik iletim (0–1)
    float           L_path,           // Atmosfer yol radyansı [W/m²/sr]
    float           L_sky,            // Aşağıya atmosfer radyansı (yansıma için)
    float           E_solar,          // Yüzeye gelen güneş enerjisi [W/m²]
    float           cos_sun,          // Güneş-yüzey açısı cosinüsü
    int             rows, int cols
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= cols || y >= rows) return;

    int idx   = y * cols + x;
    int matID = d_mat_id[idx];
    if (matID < 0 || matID >= 256) {
        d_radiance[idx] = L_path;  // Bilinmeyen materyal → sadece atmosfer
        return;
    }

    const GPUMaterialEntry& mat = d_materials[matID];
    float T_surf = d_T_surface[idx] + mat.temperature_offset_K;

    // Banta göre emisivite seç
    float eps;
    switch (band.band) {
        case 0: eps = mat.swir_eps; break;
        case 1: eps = mat.mwir_eps; break;
        default: eps = mat.lwir_eps; break;
    }

    // L_emit = ε · L_BB(T, bant) — termal emisyon
    float L_emit = bandIntegrate(T_surf, eps,
                                 band.lam_min_um, band.lam_max_um);

    // L_ref = (1 - ε) / π · L_sky_down — çevreden gelen yansıma
    float L_ref = (1.0f - eps) * L_sky / kPI;

    // SWIR için solar yansıma katkısı ekle (emisyon çok küçük)
    if (band.band == 0) {
        float L_solar_ref = swirReflectedRadiance(eps, E_solar, cos_sun);
        L_ref += L_solar_ref;
    }

    // Sensöre ulaşan toplam radyans:
    // L_sensor = τ_atm · (L_emit + L_ref) + L_path
    float L_total = tau_atm * (L_emit + L_ref) + L_path;

    d_radiance[idx] = L_total;
}

// ─── HOST yardımcı: GPU materyal tablosunu yükle ─────────────────────────────
extern "C" void uploadMaterialTableToGPU(const GPUMaterialEntry* host_table,
                                          int count) {
    cudaMemcpyToSymbol(d_materials, host_table,
                       count * sizeof(GPUMaterialEntry));
}

// ─── HOST yardımcı: Kernel launcher ─────────────────────────────────────────
extern "C" void launchPlanckKernel(
    const float* d_T_surface,
    const int*   d_mat_id,
    float*       d_radiance,
    BandParams   band,
    float T_ambient_K, float tau_atm, float L_path,
    float L_sky, float E_solar, float cos_sun,
    int rows, int cols
) {
    dim3 block(16, 16);
    dim3 grid((cols + 15) / 16, (rows + 15) / 16);
    planckRadianceKernel<<<grid, block>>>(
        d_T_surface, d_mat_id, d_radiance, band,
        T_ambient_K, tau_atm, L_path, L_sky, E_solar, cos_sun,
        rows, cols
    );
}

// ─── HOST-side doğrulama fonksiyonu (test için) ──────────────────────────────
extern "C" float planck_host(float T, float lambda_um) {
    float lambda_m = lambda_um * 1e-6f;
    if (T <= 0.0f || lambda_m <= 0.0f) return 0.0f;
    double h  = 6.62607015e-34;
    double c  = 2.99792458e+8;
    double kb = 1.380649e-23;
    double c1 = 2.0 * h * c * c;
    double c2 = (h * c) / (kb * T);
    return (float)((c1 / pow(lambda_m, 5.0)) / (exp(c2 / lambda_m) - 1.0));
}

// Güneş yükseklik açısı modeli (basit sinüsoidal gün modeli)
// hour: 0–24, 0=gece yarısı, 12=öğle
// Dönüş: yüzeye düşen güneş ışıması [W/m²], gece → 0
extern "C" float solar_irradiance_host(float hour) {
    // Güneş yükseklik açısı: 6h (doğuş) ile 18h (batış) arasında sinüsoidal
    float t = (hour - 6.0f) / 12.0f;   // 0=doğuş, 0.5=öğle, 1=batış
    if (t <= 0.0f || t >= 1.0f) return 0.0f;
    float sin_el = sinf(t * 3.14159265f);  // sin(pi*t) → tepe öğle'de
    float E_sol  = 1361.0f * sin_el;       // AM0 = 1361 W/m²
    // Atmosfer geçirgenliği yaklaşımı (AM1 ~ 1000 W/m² öğle'de)
    return fmaxf(0.0f, E_sol * 0.73f);
}

extern "C" float band_radiance_host(float T, float eps,
                                     float lam_min_um, float lam_max_um,
                                     int N) {
    float lam_min = lam_min_um * 1e-6f;
    float lam_max = lam_max_um * 1e-6f;
    float dlam = (lam_max - lam_min) / (N - 1);
    float sum = 0.0f;
    for (int i = 0; i < N; i++) {
        float lam = lam_min + i * dlam;
        float L   = eps * planck_host(lam_min_um + i*(lam_max_um-lam_min_um)/(N-1), 1.0f);
        // Yeniden doğru hesap:
        float lambda_m = lam;
        double h=6.62607015e-34, c_=2.99792458e+8, kb=1.380649e-23;
        double c1=2.0*h*c_*c_;
        double c2=(h*c_)/(kb*(double)T);
        double val = eps * (c1/pow(lambda_m,5.0)) / (exp(c2/lambda_m)-1.0);
        L = (float)val;

        float coeff = (i == 0 || i == N - 1) ? 1.0f :
                      (i % 2 == 1)           ? 4.0f : 2.0f;
        sum += coeff * L;
    }
    return sum * dlam / 3.0f;
}
