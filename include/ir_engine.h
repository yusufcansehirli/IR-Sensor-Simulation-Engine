#pragma once
#include <string>
#include "sensor_params.h"
#include "material_db.h"

// Forward declarations
struct GLFWwindow;
typedef unsigned int GLuint;

// ─────────────────────────────────────────────────────────────────────────────
// ir_engine.h  —  Ana IR Simülasyon Engine API
// Mimari belge §9.1 ile birebir örtüşür.
// ─────────────────────────────────────────────────────────────────────────────

struct glm_vec3 { float x, y, z; };

class IREngine {
public:
    // ── Başlatma konfigürasyonu ─────────────────────────────────────────────────
    struct Config {
        int width, height;
        int band_flags;               // bitmask: SWIR=1, MWIR=2, LWIR=4
        std::string material_db_path;
        std::string atm_lut_path;     // boş → analitik Beer-Lambert kullan
        bool enable_thermal_solver;
        int  thermal_update_hz;       // thermal solver güncelleme sıklığı
    };

    // ── Frame parametreleri ──────────────────────────────────────────────────────
    struct FrameParams {
        float sim_time_hours;         // 0–24 (gün içi saat)
        float range_km;               // Hedeften kamera mesafesi [km]
        float T_air_K;                // Hava sıcaklığı [K]
        float humidity_pct;           // Bağıl nem [%]
        float wind_speed_ms;          // Rüzgar hızı [m/s] (konveksiyon)
        glm_vec3 sun_direction;       // Normalize güneş yönü
        int   active_band;            // SensorParams::Band
    };

    IREngine()  = default;
    ~IREngine() = default;

    // ── Yaşam döngüsü ───────────────────────────────────────────────────────────
    bool init(const Config& cfg);
    void shutdown();

    // ── Sensör yapılandırması ────────────────────────────────────────────────────
    void setSensorParams(SensorParams::Band band, const SensorParams& params);

    // ── G-Buffer bağlantısı (OpenGL texture ID'leri) ────────────────────────────
    void updateGBuffer(GLuint pos_tex, GLuint norm_tex, GLuint mat_id_tex);

    // ── Ana render ──────────────────────────────────────────────────────────────
    void renderFrame(const FrameParams& params);

    // ── Çıkışlar ────────────────────────────────────────────────────────────────
    // Ekran texture'ı (colormap uygulanmış, 8-bit RGBA)
    GLuint getOutputTexture(SensorParams::Band band) const;

    // Ham radyans tamponu — HIL için (CPU'ya kopyalar, float32, W/m²/sr)
    void getRadianceBuffer(SensorParams::Band band, float* out_host) const;

    // ── Durum sorgusu ────────────────────────────────────────────────────────────
    bool isInitialized() const { return initialized_; }
    int  width()         const { return width_; }
    int  height()        const { return height_; }

    // ── Renklendirme ─────────────────────────────────────────────────────────────
    enum Colormap { GRAYSCALE = 0, BLACKHOT = 1, WHITEHOT = 2, RAINBOW = 3 };
    void setColormap(Colormap cm) { colormap_ = cm; }
    void setGainLevel(float gain, float level) { gain_ = gain; level_ = level; }

private:
    // ── Dahili pipeline adımları ─────────────────────────────────────────────────
    void stepThermalSolver(float sim_time_hours, float T_air_K,
                           float wind_speed_ms, const glm_vec3& sun_dir);
    void computeRadiance(const FrameParams& p);
    void applySensorModel(SensorParams::Band band);
    void displayBand(SensorParams::Band band);

    // ── GPU bellek yönetimi ──────────────────────────────────────────────────────
    void allocateBuffers();
    void freeBuffers();
    void uploadMaterialDB();

    // ── Durum ───────────────────────────────────────────────────────────────────
    bool  initialized_ = false;
    int   width_ = 0, height_ = 0;
    int   band_flags_ = 0;
    int   frame_count_ = 0;
    int   thermal_update_interval_ = 1;

    Colormap colormap_ = WHITEHOT;
    float    gain_     = 1.0f;
    float    level_    = 0.5f;

    SensorParams sensor_[3];            // indexed by Band enum
    MaterialDB   materialDB_;

    // ── GPU 버퍼lar (CUDA device pointer) ──────────────────────────────────────
    float*    d_T_surface_    = nullptr; // Yüzey sıcaklıkları [K]
    float*    d_radiance_[3]  = {};      // Radyans bufferları (SWIR, MWIR, LWIR)
    uint16_t* d_sensor_out_[3]= {};      // Sensör çıkışları (uint16)
    float*    d_float_display_[3]= {};   // Normalize float display buffer (0..1)
    float*    d_fpn_map_[3]   = {};      // Fixed Pattern Noise haritaları
    void*     d_rand_states_  = nullptr; // curand states

    // GPU materyal LUT
    void*     d_material_lut_ = nullptr;

    // ── OpenGL kaynakları ────────────────────────────────────────────────────────
    GLuint    fbo_gbuffer_  = 0;
    GLuint    tex_pos_      = 0, tex_norm_ = 0, tex_mat_ = 0;
    GLuint    tex_output_[3]= {};        // Display texture (SWIR, MWIR, LWIR)
    GLuint    vao_quad_     = 0, vbo_quad_ = 0;
    GLuint    prog_gbuf_    = 0, prog_display_ = 0;

    // Terrain resources
    GLuint    vao_terrain_  = 0;
    GLuint    vbo_terrain_  = 0;
    GLuint    ebo_terrain_  = 0;
    int       terrain_index_count_ = 0;

    // CUDA-OpenGL interop kaynakları
    void*     cuda_tex_resource_[3] = {};

    // Atmosfer LUT CUDA texture
    void*     tex_atm_tau_  = nullptr;
    void*     tex_atm_lpath_= nullptr;
};
