#pragma once
#include <cstdint>

// ─────────────────────────────────────────────────────────────────────────────
// sensor_params.h  —  IR Sensör Parametreleri
// Mimari belge §5.2 ile birebir örtüşür.
// ─────────────────────────────────────────────────────────────────────────────

struct SensorParams {
    // ── Bant Seçimi ────────────────────────────────────────────────────────────
    enum Band : int { SWIR = 0, MWIR = 1, LWIR = 2 };
    Band band;

    // ── Spektral parametreler ───────────────────────────────────────────────────
    float lambda_center_um;   // Bant merkezi [μm]
    float lambda_min_um;      // Minimum dalga boyu [μm]
    float lambda_max_um;      // Maksimum dalga boyu [μm]

    // ── Optik ──────────────────────────────────────────────────────────────────
    float focal_length_mm;
    float aperture_diameter_mm;
    float f_number;            // f/# = focal_length / aperture
    float pixel_pitch_um;      // Dedektör piksel boyutu [μm]
    float blur_spot_mrad;      // Küresel aberasyon / odak hatası [mrad]

    // ── FPA / Dedektör ─────────────────────────────────────────────────────────
    int   fpa_rows;
    int   fpa_cols;
    float integration_time_ms;
    float d_star;             // D* [cm·Hz^½/W] — dedektör kalite faktörü
    float nedt_mk;            // NEdT [mK] — temporal noise eşdeğer sıcaklık
    float fpn_sigma_fraction; // FPN piksel sapması / ortalama sinyal oranı

    // ── AGC ────────────────────────────────────────────────────────────────────
    bool  agc_enabled;
    float agc_level;          // 0.0–1.0, histogram percentile
    float manual_gain;
    float manual_level;

    // ── Bloom / Halasyon ───────────────────────────────────────────────────────
    float bloom_threshold;      // Bu radyans değerinin üstü taşar [W/m²/sr]
    float bloom_spread_pixels;  // Taşma yarıçapı [piksel]
};

// ─── Hazır konfigürasyonlar ────────────────────────────────────────────────────

// MWIR kamera — WAT MSA240G26C1C1N4001 benzeri
inline SensorParams makeMWIR() {
    return SensorParams{
        .band                   = SensorParams::MWIR,
        .lambda_center_um       = 4.0f,
        .lambda_min_um          = 3.5f,
        .lambda_max_um          = 4.9f,
        .focal_length_mm        = 100.0f,
        .aperture_diameter_mm   = 50.0f,
        .f_number               = 2.0f,
        .pixel_pitch_um         = 15.0f,
        .blur_spot_mrad         = 0.05f,
        .fpa_rows               = 512,
        .fpa_cols               = 640,
        .integration_time_ms    = 10.0f,
        .d_star                 = 3e10f,
        .nedt_mk                = 25.0f,
        .fpn_sigma_fraction     = 0.02f,
        .agc_enabled            = true,
        .agc_level              = 0.98f,
        .manual_gain            = 1.0f,
        .manual_level           = 0.5f,
        .bloom_threshold        = 1e-2f,
        .bloom_spread_pixels    = 3.0f,
    };
}

// LWIR kamera — genel broadband LWIR
inline SensorParams makeLWIR() {
    return SensorParams{
        .band                   = SensorParams::LWIR,
        .lambda_center_um       = 10.0f,
        .lambda_min_um          = 8.5f,
        .lambda_max_um          = 11.5f,
        .focal_length_mm        = 75.0f,
        .aperture_diameter_mm   = 37.5f,
        .f_number               = 2.0f,
        .pixel_pitch_um         = 17.0f,
        .blur_spot_mrad         = 0.04f,
        .fpa_rows               = 480,
        .fpa_cols               = 640,
        .integration_time_ms    = 12.0f,
        .d_star                 = 1e10f,
        .nedt_mk                = 50.0f,
        .fpn_sigma_fraction     = 0.015f,
        .agc_enabled            = true,
        .agc_level              = 0.98f,
        .manual_gain            = 1.0f,
        .manual_level           = 0.5f,
        .bloom_threshold        = 5e-3f,
        .bloom_spread_pixels    = 2.0f,
    };
}

// SWIR kamera — InGaAs sensör
inline SensorParams makeSWIR() {
    return SensorParams{
        .band                   = SensorParams::SWIR,
        .lambda_center_um       = 1.55f,
        .lambda_min_um          = 1.0f,
        .lambda_max_um          = 1.7f,
        .focal_length_mm        = 50.0f,
        .aperture_diameter_mm   = 25.0f,
        .f_number               = 2.0f,
        .pixel_pitch_um         = 5.0f,
        .blur_spot_mrad         = 0.02f,
        .fpa_rows               = 1024,
        .fpa_cols               = 1280,
        .integration_time_ms    = 5.0f,
        .d_star                 = 5e12f,
        .nedt_mk                = 10.0f,
        .fpn_sigma_fraction     = 0.01f,
        .agc_enabled            = true,
        .agc_level              = 0.99f,
        .manual_gain            = 1.0f,
        .manual_level           = 0.5f,
        .bloom_threshold        = 0.1f,
        .bloom_spread_pixels    = 5.0f,
    };
}
