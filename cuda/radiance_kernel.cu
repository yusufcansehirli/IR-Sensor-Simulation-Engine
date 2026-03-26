// radiance_kernel.cu
// ─────────────────────────────────────────────────────────────────────────────
// Radyans birleştirme — G-Buffer verilerinden tam pipeline hesabı.
// Planck + atmosfer + yansıma birleştirir.
// Mimari belge §6.1 (Adım 3) ile örtüşür.
// ─────────────────────────────────────────────────────────────────────────────
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "gpu_types.cuh"

// ─── AtmResult ve extern __device__ fonksiyon bildirimleri ───────────────────
// atm_lut.cu'daki device fonksiyonları — CUDA separable compilation ile bağlanır
struct AtmResult {
    float tau[3];    // iletim: SWIR, MWIR, LWIR
    float Lpath[3];  // yol radyansı
    float Lsky;      // aşağı atmosfer radyansı
};

extern __device__ AtmResult computeAtmosphere(
    float range_km, float T_air, float humidity,
    cudaTextureObject_t tex_tau,
    cudaTextureObject_t tex_lpath,
    bool use_lut
);

// ─── Radyans kernel giriş parametreleri ─────────────────────────────────────
struct RadianceParams {
    float range_km;
    float T_air_K;
    float humidity_pct;
    float sim_time_hours;
    float sun_el;           // güneş yükseklik açısı [radyan]
    float sun_az;           // güneş azimut açısı [radyan]
    bool  use_lut;
};

// Bant için çıkış bufferı seçici
struct BandSelectParams {
    float lam_min_um;
    float lam_max_um;
    int   band_idx;         // 0=SWIR, 1=MWIR, 2=LWIR
};

// ─── Planck device fonksiyonu (forward declaration) ──────────────────────────
__device__ float bandIntegrate(float T, float eps,
                                float lam_min_um, float lam_max_um,
                                int N = 7);

__device__ float swirReflectedRadiance(float eps, float E_solar, float cos_sun);

// GPU materyal tablosu (planck_kernel.cu'dan paylaşımlı)
__constant__ extern GPUMaterialEntry d_materials[256];

#ifndef M_PI
#define M_PI 3.14159265358979f
#endif

__global__ void radianceCombineKernel(
    // G-Buffer girişleri
    const float4*  d_world_pos,    // (x,y,z,_) dünya koordinatları
    const float4*  d_world_normal, // (nx,ny,nz,_)
    const int*     d_mat_id,       // Materyal ID
    // Fizik verileri
    const float*   d_T_surface,    // Yüzey sıcaklıkları [K]
    // Atmosfer texture (0 → analitik)
    cudaTextureObject_t tex_tau,
    cudaTextureObject_t tex_lpath,
    // Parametreler
    RadianceParams  atm,
    BandSelectParams band,
    // Çıkış
    float*         d_radiance,     // W/m²/sr
    int rows, int cols
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= cols || y >= rows) return;

    int idx   = y * cols + x;
    int matID = d_mat_id[idx];

    // Arka plan piksel (matID = -1) → 0 veya gök radyansı
    if (matID < 0 || matID >= 256) {
        // Gökyüzü pikseli — basit mavi veya atmosfer radyansı
        d_radiance[idx] = 0.0f;
        return;
    }

    const GPUMaterialEntry& mat = d_materials[matID];

    // ── Atmosferik iletim hesabı ─────────────────────────────────────────────
    AtmResult atm_r = computeAtmosphere(
        atm.range_km, atm.T_air_K, atm.humidity_pct,
        tex_tau, tex_lpath, atm.use_lut
    );

    float tau   = atm_r.tau[band.band_idx];
    float L_path= atm_r.Lpath[band.band_idx];
    float L_sky = atm_r.Lsky;

    // ── Banta göre emisivite ─────────────────────────────────────────────────
    float eps;
    switch (band.band_idx) {
        case 0: eps = mat.swir_eps; break;
        case 1: eps = mat.mwir_eps; break;
        default: eps = mat.lwir_eps; break;
    }

    // ── Yüzey sıcaklığı ─────────────────────────────────────────────────────
    float T_surf = d_T_surface[idx] + mat.temperature_offset_K;

    // ── Termal emisyon ───────────────────────────────────────────────────────
    float L_emit = bandIntegrate(T_surf, eps, band.lam_min_um, band.lam_max_um);

    // ── Yansıma ─────────────────────────────────────────────────────────────
    // BRDF basitleştirme: Lambertian (diffuse)
    float L_ref = (1.0f - eps) * L_sky / M_PI;

    // SWIR solar katkısı
    if (band.band_idx == 0) {
        float4 normal = d_world_normal[idx];
        float3 sun_dir = make_float3(cosf(atm.sun_el) * sinf(atm.sun_az),
                                     cosf(atm.sun_el) * cosf(atm.sun_az),
                                     sinf(atm.sun_el));
        float cos_sun = fmaxf(normal.x * sun_dir.x +
                               normal.y * sun_dir.y +
                               normal.z * sun_dir.z, 0.0f);
        float E_sol = 1000.0f * fmaxf(sinf(atm.sun_el), 0.0f);
        L_ref      += (1.0f - eps) * E_sol * cos_sun / M_PI;
    }

    // ── Toplam sensör radyansı ───────────────────────────────────────────────
    d_radiance[idx] = tau * (L_emit + L_ref) + L_path;
}

// ─── HOST launcher ────────────────────────────────────────────────────────────
extern "C" void launchRadianceCombine(
    const float4* d_world_pos,
    const float4* d_world_normal,
    const int*    d_mat_id,
    const float*  d_T_surface,
    cudaTextureObject_t tex_tau,
    cudaTextureObject_t tex_lpath,
    RadianceParams  atm,
    BandSelectParams band,
    float* d_radiance,
    int rows, int cols
) {
    dim3 block(16, 16);
    dim3 grid((cols + 15) / 16, (rows + 15) / 16);
    radianceCombineKernel<<<grid, block>>>(
        d_world_pos, d_world_normal, d_mat_id, d_T_surface,
        tex_tau, tex_lpath,
        atm, band, d_radiance,
        rows, cols
    );
}
