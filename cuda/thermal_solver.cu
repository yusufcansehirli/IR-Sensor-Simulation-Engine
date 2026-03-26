// thermal_solver.cu
// ─────────────────────────────────────────────────────────────────────────────
// Yüzey termal durumu çözücüsü — basit ısı dengesi + diurnal güneş döngüsü.
// Mimari belge §2.4 ile örtüşür.
//
// Cₚ · dT/dt = Q_solar·α - Q_conv - Q_rad_emit + Q_conduction
// Euler adımı ile sayısal integrasyon (gerçek zamanlı simülasyonda yeterli)
// ─────────────────────────────────────────────────────────────────────────────
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979f
#endif

// ─── Sabitleri ────────────────────────────────────────────────────────────────
__constant__ constexpr float kSB = 5.670374419e-8f;  // Stefan-Boltzmann [W/m²/K⁴]

// ─── Güneş irradyansı — diurnal model ────────────────────────────────────────
// hour: [0,24]  sunrise ≈ 6, sunset ≈ 18
// Dönüş: yüzeye gelen global horizontal irradyans [W/m²]
__device__ __forceinline__ float solarIrradiance(float hour,
                                                   float latitude_deg = 35.0f) {
    // Basit sinüsoidal model
    float t_solar = hour - 12.0f;                      // -12..12
    float cos_za  = cosf(t_solar * M_PI / 12.0f);      // zenit açısı cosinüsü
    cos_za = fmaxf(cos_za, 0.0f);                       // gece → 0
    return 1000.0f * cos_za;                            // max ~1000 W/m² (atmosfer)
}

// Güneş yön vektörü (normalize)
__device__ float3 sunDirection(float hour) {
    float t_solar = hour - 12.0f;
    float az = M_PI * t_solar / 12.0f;     // azimuth (basit)
    float el = M_PI_2f - fabsf(az);         // elevation
    el = fmaxf(el, 0.0f);
    return make_float3(sinf(az) * cosf(el),
                       cosf(az) * cosf(el),
                       sinf(el));
}

// ─────────────────────────────────────────────────────────────────────────────
// KERNEL: Yüzey sıcaklığı güncelleme (Euler adımı)
// ─────────────────────────────────────────────────────────────────────────────
struct ThermalParams {
    float T_air_K;          // Ortam hava sıcaklığı [K]
    float wind_speed_ms;    // Rüzgar hızı [m/s]
    float sim_time_hours;   // Simülasyon saati [0–24]
    float dt_seconds;       // Adım boyutu [s]
};

// Materyal termal özellikleri (GPU'da flat diziden okunur)
struct ThermalMaterial {
    float thermal_mass;          // J/(m³·K)
    float solar_absorptivity;    // 0–1
    float lwir_eps;              // Uzun dalga emisivite (radyatif kayıp)
    float temperature_offset_K;  // Sabit sıcaklık ofseti (motor, egzoz)
};

__global__ void thermalSolverKernel(
    float*               d_T_surface,   // (In/Out) Yüzey sıcaklıkları [K]
    const int*           d_mat_id,      // Materyal ID haritası
    const ThermalMaterial* d_mat_table, // Materyal termal özellikleri (256 entry)
    const float3*        d_normals,     // Yüzey normalleri
    ThermalParams        params,
    int rows, int cols
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= cols || y >= rows) return;

    int idx   = y * cols + x;
    int matID = d_mat_id[idx];
    if (matID < 0 || matID >= 256) return;

    const ThermalMaterial& mat = d_mat_table[matID];

    // Sabit ofsetli materyaller — solver çalıştırma (motor, egzoz)
    if (mat.temperature_offset_K > 50.0f) {
        // Sıcaklığı çevre + ofset olarak sabitle
        d_T_surface[idx] = params.T_air_K + mat.temperature_offset_K;
        return;
    }

    float T     = d_T_surface[idx];
    float T_air = params.T_air_K;

    // ── Güneş irradyansı ─────────────────────────────────────────────────────
    float E_sol   = solarIrradiance(params.sim_time_hours);
    float3 sun    = sunDirection(params.sim_time_hours);
    float3 normal = d_normals[idx];

    // Yüzey-güneş açısı (Lambertian projeksiyon)
    float cos_sun = fmaxf(normal.x * sun.x +
                           normal.y * sun.y +
                           normal.z * sun.z, 0.0f);

    // Q_solar = α · E_sol · cos(θ)
    float Q_solar = mat.solar_absorptivity * E_sol * cos_sun;

    // ── Konvektif kayıp ───────────────────────────────────────────────────────
    // Newton soğuma yasası
    // h_conv: [W/(m²·K)] — rüzgar hızına bağlı yaklaşım (nükleer)
    // h = 5.7 + 3.8 * v  (deneysel, düz yüzey)
    float h_conv  = 5.7f + 3.8f * params.wind_speed_ms;
    float Q_conv  = h_conv * (T - T_air);

    // ── Radyatif kayıp ────────────────────────────────────────────────────────
    // Q_rad = ε · σ · T⁴ — (ε_sky · σ · T_sky⁴) gelen net
    float T_sky   = T_air - 20.0f;  // Etkili gökyüzü sıcaklığı (basit model)
    float Q_rad   = mat.lwir_eps * kSB * (T * T * T * T - T_sky * T_sky * T_sky * T_sky);

    // ── Isı denklemi: Euler adımı ─────────────────────────────────────────────
    // Cₚ · dT/dt = Q_solar - Q_conv - Q_rad
    // dT/dt = (Q_solar - Q_conv - Q_rad) / thermal_mass
    float layer_thickness = 0.05f;            // efektif ısı derinliği [m] — 5 cm
    float Cp_vol = mat.thermal_mass;          // J/(m³·K)
    float dTdt = (Q_solar - Q_conv - Q_rad) / (Cp_vol * layer_thickness);

    float T_new = T + dTdt * params.dt_seconds;

    // Fiziksel sınırlar: aşırı ısınma veya donmayı engelle
    T_new = fmaxf(T_new, 223.0f);   // -50°C minimum
    T_new = fminf(T_new, 450.0f);   // 177°C maksimum (yapay yüzey)

    // motor_housing ve küçük ofsetler için ofset ekle
    T_new += mat.temperature_offset_K;
    T_new = fmaxf(T_new, T_air - 10.0f);  // çok soğuk olmayı engelle

    d_T_surface[idx] = T_new;
}

// ─── T_surface'i başlat (gündoğumu başına ambient value) ────────────────────
__global__ void initThermalKernel(
    float* d_T_surface,
    const int* d_mat_id,
    const ThermalMaterial* d_mat_table,
    float T_ambient_K,
    int rows, int cols
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= cols || y >= rows) return;

    int idx   = y * cols + x;
    int matID = d_mat_id[idx];
    float T_offset = (matID >= 0 && matID < 256) ?
                     d_mat_table[matID].temperature_offset_K : 0.0f;

    d_T_surface[idx] = T_ambient_K + T_offset;
}

// ─── HOST launcher ────────────────────────────────────────────────────────────
extern "C" void launchThermalSolver(
    float* d_T_surface,
    const int* d_mat_id,
    const ThermalMaterial* d_mat_table,
    const float3* d_normals,
    ThermalParams params,
    int rows, int cols
) {
    dim3 block(16, 16);
    dim3 grid((cols + 15) / 16, (rows + 15) / 16);
    thermalSolverKernel<<<grid, block>>>(
        d_T_surface, d_mat_id, d_mat_table, d_normals, params, rows, cols
    );
}

extern "C" void launchThermalInit(
    float* d_T_surface,
    const int* d_mat_id,
    const ThermalMaterial* d_mat_table,
    float T_ambient_K,
    int rows, int cols
) {
    dim3 block(16, 16);
    dim3 grid((cols + 15) / 16, (rows + 15) / 16);
    initThermalKernel<<<grid, block>>>(
        d_T_surface, d_mat_id, d_mat_table, T_ambient_K, rows, cols
    );
}

// solar_irradiance_host planck_kernel.cu'da tanımlı (canonical)
