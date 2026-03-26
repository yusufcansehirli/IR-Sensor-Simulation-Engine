// gpu_types.cuh
// ─────────────────────────────────────────────────────────────────────────────
// Tüm CUDA kernel dosyaları arasında paylaşılan GPU struct tanımları.
// Bu başlık hem .cu hem de .cpp dosyalarında include edilebilir.
// ─────────────────────────────────────────────────────────────────────────────
#pragma once
#include <cuda_runtime.h>

// ─── Materyal tablosu ─────────────────────────────────────────────────────────
// 256 kayıt, constant memory'de tutulur.
struct GPUMaterialEntry {
    float swir_eps;              // SWIR emisivitesi  (1.0–2.5 μm)
    float mwir_eps;              // MWIR emisivitesi  (3.5–5.0 μm)
    float lwir_eps;              // LWIR emisivitesi  (8.0–12 μm)
    float thermal_mass;          // ısıl kütle [J/m²/K]
    float solar_absorptivity;    // güneş soğurma katsayısı
    float temperature_offset_K;  // G-Buffer sıcaklık ofseti [K]
    float _pad[2];               // 32-byte hizalama
};

// ─── Planck bant parametreleri ────────────────────────────────────────────────
struct BandParams {
    int   band;          // 0=SWIR, 1=MWIR, 2=LWIR
    float lam_min_um;    // bant alt sınırı [μm]
    float lam_max_um;    // bant üst sınırı [μm]
    float lam_center_um; // bant merkezi [μm]
};

// ─── Sensör FPA GPU parametreleri ────────────────────────────────────────────
struct SensorParamsGPU {
    float f_number;
    float pixel_pitch_um;
    float integration_time_ms;
    float d_star;               // dedektörlük [cm √Hz / W]
    float nedt_mk;              // eşdeğer gürültü sıcaklığı farkı [mK]
    float fpn_sigma_fraction;   // FPN sigma / tam scala oranı
    float bloom_threshold;      // bloom eşiği (normalize)
    float bloom_spread_pixels;  // bloom yayılma pikseli
    float agc_gain;
    float agc_level;
    int   band;
};
