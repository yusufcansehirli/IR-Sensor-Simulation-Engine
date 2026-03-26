#pragma once

// ─────────────────────────────────────────────────────────────────────────────
// ir_sensor_plugin.h  —  C API for Unity / Unreal Engine 5
// Mimari belge §7.2 ile birebir örtüşür.
// ─────────────────────────────────────────────────────────────────────────────

#ifdef _WIN32
  #define IR_EXPORT __declspec(dllexport)
#else
  #define IR_EXPORT __attribute__((visibility("default")))
#endif

#include "sensor_params.h"

extern "C" {

// ── Yaşam döngüsü ─────────────────────────────────────────────────────────────
// band_flags: bit0=SWIR, bit1=MWIR, bit2=LWIR
IR_EXPORT int  ir_init(int width, int height, int band_flags);
IR_EXPORT void ir_shutdown();

// ── Veri yükleme ──────────────────────────────────────────────────────────────
IR_EXPORT int  ir_load_material_db(const char* yaml_path);
IR_EXPORT int  ir_load_atm_lut(const char* h5_path);    // HAVE_HDF5 gerektirir

// ── Sensör konfigürasyonu ─────────────────────────────────────────────────────
IR_EXPORT void ir_set_sensor_params(int band, const SensorParams* params);

// ── G-Buffer (Unity/UE5 native texture handle) ───────────────────────────────
IR_EXPORT void ir_set_gbuffer(void* position_tex,
                               void* normal_tex,
                               void* material_id_tex);

// ── Render ───────────────────────────────────────────────────────────────────
IR_EXPORT void ir_render_frame(float sim_time_hours,
                                float range_km,
                                float T_air_K,
                                float humidity_pct,
                                int   active_band);

// ── Çıkış texture handle (Unity RenderTexture'a bağlanır) ────────────────────
IR_EXPORT void* ir_get_output_texture(int band);

// ── Ham radyans (HIL çıkışı) ─────────────────────────────────────────────────
// out_buffer: float[width * height] — çağıran taraf allocate eder
IR_EXPORT void ir_get_radiance_buffer(int band, float* out_buffer);

// ── Görüntü ayarları ─────────────────────────────────────────────────────────
// colormap: 0=grayscale, 1=blackhot, 2=whitehot, 3=rainbow
IR_EXPORT void ir_set_colormap(int colormap);
IR_EXPORT void ir_set_gain_level(float gain, float level);

} // extern "C"
