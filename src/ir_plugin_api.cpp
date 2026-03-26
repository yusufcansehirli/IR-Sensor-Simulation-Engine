// ir_plugin_api.cpp
// ─────────────────────────────────────────────────────────────────────────────
// Unity / Unreal Engine 5 C API implementasyonu.
// Mimari belge §7.2 ile örtüşür.
// ─────────────────────────────────────────────────────────────────────────────
#include "ir_sensor_plugin.h"
#include "ir_engine.h"
#include <cstdio>

static IREngine* g_engine = nullptr;

int ir_init(int width, int height, int band_flags) {
    if (g_engine) {
        fprintf(stderr, "[Plugin] ir_init: önceki engine kapatılıyor\n");
        g_engine->shutdown();
        delete g_engine;
    }
    g_engine = new IREngine();
    IREngine::Config cfg{};
    cfg.width               = width;
    cfg.height              = height;
    cfg.band_flags          = band_flags;
    cfg.material_db_path    = "data/material_database.yaml";
    cfg.enable_thermal_solver = true;
    cfg.thermal_update_hz   = 10;
    return g_engine->init(cfg) ? 0 : -1;
}

void ir_shutdown() {
    if (g_engine) {
        g_engine->shutdown();
        delete g_engine;
        g_engine = nullptr;
    }
}

int ir_load_material_db(const char* yaml_path) {
    // Engine yeniden başlatılmadan materyal DB değiştirme
    // Şimdilik: engine'i sil ve yeni konfigürasyonla başlat
    fprintf(stderr, "[Plugin] ir_load_material_db: engine restart gerekiyor\n");
    return -1;
}

int ir_load_atm_lut(const char* h5_path) {
    // atm_lut.cu'daki yükleme fonksiyonunu çağır
    extern bool atm_load_lut(const char*);
    return atm_load_lut(h5_path) ? 0 : -1;
}

void ir_set_sensor_params(int band, const SensorParams* params) {
    if (!g_engine || !params) return;
    g_engine->setSensorParams((SensorParams::Band)band, *params);
}

void ir_set_gbuffer(void* position_tex, void* normal_tex, void* material_id_tex) {
    if (!g_engine) return;
    // DX11/Vulkan handle → OpenGL interop (platform-specific)
    // Linux OpenGL modunda direkt GLuint kullan
    g_engine->updateGBuffer(
        (GLuint)(uintptr_t)position_tex,
        (GLuint)(uintptr_t)normal_tex,
        (GLuint)(uintptr_t)material_id_tex
    );
}

void ir_render_frame(float sim_time_hours, float range_km,
                     float T_air_K, float humidity_pct, int active_band) {
    if (!g_engine) return;
    IREngine::FrameParams fp{};
    fp.sim_time_hours = sim_time_hours;
    fp.range_km       = range_km;
    fp.T_air_K        = T_air_K;
    fp.humidity_pct   = humidity_pct;
    fp.wind_speed_ms  = 3.0f;   // varsayılan
    fp.sun_direction  = {0.0f, 0.707f, 0.707f};   // güneş 45° SE
    fp.active_band    = active_band;
    g_engine->renderFrame(fp);
}

void* ir_get_output_texture(int band) {
    if (!g_engine) return nullptr;
    GLuint tex = g_engine->getOutputTexture((SensorParams::Band)band);
    return (void*)(uintptr_t)tex;
}

void ir_get_radiance_buffer(int band, float* out_buffer) {
    if (!g_engine || !out_buffer) return;
    g_engine->getRadianceBuffer((SensorParams::Band)band, out_buffer);
}

void ir_set_colormap(int colormap) {
    if (!g_engine) return;
    g_engine->setColormap((IREngine::Colormap)colormap);
}

void ir_set_gain_level(float gain, float level) {
    if (!g_engine) return;
    g_engine->setGainLevel(gain, level);
}
