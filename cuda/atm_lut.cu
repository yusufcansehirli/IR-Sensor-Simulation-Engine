// atm_lut.cu
// ─────────────────────────────────────────────────────────────────────────────
// Atmosferik iletim ve yol radyansı hesabı.
// İki mod:
//   1. 3D LUT (HDF5'ten yüklenen libRadtran verisi, HAVE_HDF5 gerektirir)
//   2. Analitik Beer-Lambert fallback (libRadtran olmadan çalışır)
//
// Mimari belge §4 ile örtüşür.
// ─────────────────────────────────────────────────────────────────────────────
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>
#include <functional>

#ifdef HAVE_HDF5
#include <hdf5.h>
#endif

// ─── LUT boyutları (mimari belge §4.1) ───────────────────────────────────────
constexpr int LUT_RANGE_STEPS  = 32;   // 0.1–10 km log-scale
constexpr int LUT_TEMP_STEPS   = 16;   // 250–320 K
constexpr int LUT_HUMID_STEPS  = 16;   // 10–95 %
constexpr float LUT_RANGE_MIN  = 0.1f; // km
constexpr float LUT_RANGE_MAX  = 10.0f;
constexpr float LUT_TEMP_MIN   = 250.0f;
constexpr float LUT_TEMP_MAX   = 320.0f;
constexpr float LUT_HUMID_MIN  = 10.0f;
constexpr float LUT_HUMID_MAX  = 95.0f;

// ─── CUDA 3D texture (LUT tabanlı mod) ───────────────────────────────────────
// Her texel: (tau_swir, tau_mwir, tau_lwir, sky_radiance)
static cudaTextureObject_t g_tex_tau   = 0;
static cudaTextureObject_t g_tex_lpath = 0;
static cudaArray_t         g_arr_tau   = nullptr;
static cudaArray_t         g_arr_lpath = nullptr;
static bool                g_lut_loaded = false;

// ─── Analitik Beer-Lambert katsayıları (bant ortalamaları) ───────────────────
// α [km⁻¹] — standart troposfer (deniz seviyesi, orta nem)
struct AtmCoeffs {
    float alpha;   // zayıflama katsayısı [km⁻¹]
    float L_path_per_km; // yol radyansı katsayısı [W/m²/sr/km]
};

// Yansl table — humidity/temp etkileri basit çarpanlar ile modellenir
__constant__ AtmCoeffs d_analytic_coeffs[3] = {
    { 0.10f, 1e-5f },   // SWIR  (1.6 μm) — nispeten saydam
    { 0.15f, 5e-5f },   // MWIR  (4.0 μm) — CO₂ bandı zayıflatma
    { 0.05f, 2e-4f },   // LWIR  (10 μm)  — ozon penceresi saydam
};

// ─── Device: LUT ile atmosfer örneği al ─────────────────────────────────────
__device__ float4 sampleAtmLUT_tau(cudaTextureObject_t tex,
                                    float range_km, float T_air, float humidity) {
    float u = (log10f(range_km) - log10f(LUT_RANGE_MIN)) /
              (log10f(LUT_RANGE_MAX) - log10f(LUT_RANGE_MIN));
    float v = (T_air    - LUT_TEMP_MIN)  / (LUT_TEMP_MAX  - LUT_TEMP_MIN);
    float w = (humidity - LUT_HUMID_MIN) / (LUT_HUMID_MAX - LUT_HUMID_MIN);
    u = fmaxf(0.0f, fminf(1.0f, u));
    v = fmaxf(0.0f, fminf(1.0f, v));
    w = fmaxf(0.0f, fminf(1.0f, w));
    return tex3D<float4>(tex, u, v, w);
}

// ─── Device: Analitik Beer-Lambert ───────────────────────────────────────────
__device__ void computeAtmAnalytic(
    int band, float range_km, float T_air, float humidity,
    float& tau, float& L_path, float& L_sky
) {
    // Nem etkisi: yüksek nem daha fazla zayıflama (yaklaşım)
    float humidity_factor = 1.0f + 0.5f * (humidity - 50.0f) / 50.0f;
    // Sıcaklık etkisi: küçük etki
    float temp_factor     = 1.0f + 0.02f * (T_air - 288.0f) / 30.0f;

    float alpha_eff = d_analytic_coeffs[band].alpha *
                      humidity_factor * temp_factor;
    tau   = expf(-alpha_eff * range_km);

    // Atmosferik emisyon yaklaşımı (Kirchhoff): (1 - tau) · L_BB(T_atm, band)
    // L_BB yaklaşımı: bant bağımlı sabit ile ölçekle
    float L_BB_atm = d_analytic_coeffs[band].L_path_per_km *
                     expf((T_air - 288.0f) / 30.0f);
    L_path = (1.0f - tau) * L_BB_atm * range_km;
    L_sky  = 2.0f * L_BB_atm;   // Yarı küre entegrasyonu yaklaşımı
}

// ─── Output yapısı ───────────────────────────────────────────────────────────
struct AtmResult {
    float tau[3];    // iletim: SWIR, MWIR, LWIR
    float Lpath[3];  // yol radyansı
    float Lsky;      // aşağıya atmosfer radyansı (yansıma için)
};

// ─── Device: Her piksel için atmosfer hesabı ─────────────────────────────────
__device__ AtmResult computeAtmosphere(
    float range_km, float T_air, float humidity,
    cudaTextureObject_t tex_tau,
    cudaTextureObject_t tex_lpath,
    bool use_lut
) {
    AtmResult result;
    if (use_lut) {
        float4 tau   = sampleAtmLUT_tau(tex_tau,   range_km, T_air, humidity);
        float4 lpath = sampleAtmLUT_tau(tex_lpath,  range_km, T_air, humidity);
        result.tau[0]   = tau.x;   // SWIR
        result.tau[1]   = tau.y;   // MWIR
        result.tau[2]   = tau.z;   // LWIR
        result.Lpath[0] = lpath.x;
        result.Lpath[1] = lpath.y;
        result.Lpath[2] = lpath.z;
        result.Lsky     = tau.w;   // sky downwelling
    } else {
        float tau_tmp, Lp_tmp, Lsky_tmp;
        for (int b = 0; b < 3; b++) {
            computeAtmAnalytic(b, range_km, T_air, humidity,
                               tau_tmp, Lp_tmp, Lsky_tmp);
            result.tau[b]   = tau_tmp;
            result.Lpath[b] = Lp_tmp;
        }
        // Ortalama sky radyansı
        computeAtmAnalytic(2, range_km, T_air, humidity,
                           tau_tmp, Lp_tmp, Lsky_tmp);
        result.Lsky = Lsky_tmp;
    }
    return result;
}

// ─── HOST: LUT yükleme ───────────────────────────────────────────────────────
#ifdef HAVE_HDF5
bool loadAtmLUT_HDF5(const char* path) {
    hid_t file = H5Fopen(path, H5F_ACC_RDONLY, H5P_DEFAULT);
    if (file < 0) {
        fprintf(stderr, "[AtmLUT] HDF5 dosyası açılamadı: %s\n", path);
        return false;
    }

    int N = LUT_RANGE_STEPS * LUT_TEMP_STEPS * LUT_HUMID_STEPS;
    std::vector<float> tau_swir(N), tau_mwir(N), tau_lwir(N);
    std::vector<float> Lp_swir(N),  Lp_mwir(N),  Lp_lwir(N);

    auto readDS = [&](const char* name, std::vector<float>& buf) {
        hid_t ds = H5Dopen2(file, name, H5P_DEFAULT);
        H5Dread(ds, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, buf.data());
        H5Dclose(ds);
    };

    readDS("tau_swir",   tau_swir);
    readDS("tau_mwir",   tau_mwir);
    readDS("tau_lwir",   tau_lwir);
    readDS("Lpath_swir", Lp_swir);
    readDS("Lpath_mwir", Lp_mwir);
    readDS("Lpath_lwir", Lp_lwir);
    H5Fclose(file);

    // float4 dizisine dönüştür
    std::vector<float4> tau_arr(N), lpath_arr(N);
    for (int i = 0; i < N; i++) {
        tau_arr[i]   = make_float4(tau_swir[i],  tau_mwir[i],  tau_lwir[i],  1.0f);
        lpath_arr[i] = make_float4(Lp_swir[i],   Lp_mwir[i],  Lp_lwir[i],  0.0f);
    }

    // 3D CUDA array oluştur ve texture bağla (tek bant)
    cudaChannelFormatDesc chanDesc = cudaCreateChannelDesc<float4>();
    cudaExtent extent = make_cudaExtent(LUT_RANGE_STEPS, LUT_TEMP_STEPS, LUT_HUMID_STEPS);

    auto createTex = [&](cudaArray_t& arr, cudaTextureObject_t& tex,
                          const std::vector<float4>& data) {
        cudaMalloc3DArray(&arr, &chanDesc, extent);
        cudaMemcpy3DParms cpy = {};
        cpy.srcPtr  = make_cudaPitchedPtr((void*)data.data(),
                                           LUT_RANGE_STEPS * sizeof(float4),
                                           LUT_RANGE_STEPS, LUT_TEMP_STEPS);
        cpy.dstArray = arr;
        cpy.extent   = extent;
        cpy.kind     = cudaMemcpyHostToDevice;
        cudaMemcpy3D(&cpy);

        cudaResourceDesc resDesc = {};
        resDesc.resType         = cudaResourceTypeArray;
        resDesc.res.array.array = arr;
        cudaTextureDesc texDesc = {};
        texDesc.filterMode      = cudaFilterModeLinear;
        texDesc.readMode        = cudaReadModeElementType;
        texDesc.normalizedCoords = 1;
        texDesc.addressMode[0]  = texDesc.addressMode[1] =
        texDesc.addressMode[2]  = cudaAddressModeClamp;
        cudaCreateTextureObject(&tex, &resDesc, &texDesc, nullptr);
    };

    createTex(g_arr_tau,   g_tex_tau,   tau_arr);
    createTex(g_arr_lpath, g_tex_lpath, lpath_arr);

    g_lut_loaded = true;
    printf("[AtmLUT] LUT yüklendi: %s\n", path);
    return true;
}
#endif  // HAVE_HDF5

extern "C" {
    bool atm_load_lut(const char* path) {
#ifdef HAVE_HDF5
        return loadAtmLUT_HDF5(path);
#else
        fprintf(stderr, "[AtmLUT] HDF5 desteği yok — analitik mod kullanılıyor\n");
        return false;
#endif
    }

    void atm_free_lut() {
        if (g_tex_tau)   cudaDestroyTextureObject(g_tex_tau);
        if (g_tex_lpath) cudaDestroyTextureObject(g_tex_lpath);
        if (g_arr_tau)   cudaFreeArray(g_arr_tau);
        if (g_arr_lpath) cudaFreeArray(g_arr_lpath);
        g_tex_tau = g_tex_lpath = 0;
        g_arr_tau = g_arr_lpath = nullptr;
        g_lut_loaded = false;
    }

    bool atm_is_lut_loaded() { return g_lut_loaded; }
    cudaTextureObject_t atm_get_tau_tex()   { return g_tex_tau; }
    cudaTextureObject_t atm_get_lpath_tex() { return g_tex_lpath; }
}
