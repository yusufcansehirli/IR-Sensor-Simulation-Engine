// ir_engine.cpp
// ─────────────────────────────────────────────────────────────────────────────
// IREngine — ana simülasyon döngüsü.
// OpenGL + CUDA pipeline yönetimi.
// Mimari belge §6 ve §9.1 ile örtüşür.
// ─────────────────────────────────────────────────────────────────────────────
#include "ir_engine.h"
#include "sensor_params.h"
#include "material_db.h"

#include <glad/glad.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <curand_kernel.h>
#include <cstdio>
#include <cmath>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

// ─── Forward declarations (CUDA kernelleri) ───────────────────────────────────
// BandParams — planck_kernel.cu ile sync
struct BandParams {
    int   band;
    float lam_min_um;
    float lam_max_um;
    float lam_center_um;
};

// GPU materyal girişi — material_db.cpp ile sync
struct GPUMaterialEntry {
    float swir_eps, mwir_eps, lwir_eps;
    float thermal_mass, solar_absorptivity, temperature_offset_K;
    float _pad[2];
};

struct BandSelectParams;
struct RadianceParams;
struct ThermalParams;
struct ThermalMaterial;
struct AtmResult;

// SensorParamsGPU — sensor_model.cu ile sync
struct SensorParamsGPU {
    float f_number;
    float pixel_pitch_um;
    float integration_time_ms;
    float d_star;
    float nedt_mk;
    float fpn_sigma_fraction;
    float bloom_threshold;
    float bloom_spread_pixels;
    float agc_gain;
    float agc_level;
    int   band;
};

extern "C" {
    void uploadMaterialTableToGPU(const GPUMaterialEntry* table, int count);
    void launchPlanckKernel(const float*, const int*, float*, BandParams,
                             float, float, float, float, float, float, int, int);
    void launchThermalInit(float*, const int*, const void*, float, int, int);
    void launchMTF(const float*, float*, float*, int, int,
                   float, float, float);
    void sensor_init_curand(curandState**, unsigned long long, int);
    void sensor_generate_fpn(float**, curandState*, int);
    void sensor_run(const float*, uint16_t*, curandState*, const float*,
                    SensorParamsGPU, float, float, int, int);
    bool atm_load_lut(const char*);
    void atm_free_lut();
    bool atm_is_lut_loaded();
    cudaTextureObject_t atm_get_tau_tex();
    cudaTextureObject_t atm_get_lpath_tex();
    void launchRadianceCombine(const float4*, const float4*, const int*,
                                const float*, cudaTextureObject_t,
                                cudaTextureObject_t,
                                RadianceParams, BandSelectParams,
                                float*, int, int);
}

// ─── Shader yardımcı ─────────────────────────────────────────────────────────
static std::string readFile(const std::string& path) {
    std::ifstream f(path);
    if (!f) { fprintf(stderr, "[IREngine] Shader dosyası açılamadı: %s\n", path.c_str()); return ""; }
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

static GLuint compileShader(GLenum type, const std::string& src) {
    GLuint s = glCreateShader(type);
    const char* c = src.c_str();
    glShaderSource(s, 1, &c, nullptr);
    glCompileShader(s);
    GLint ok; glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char log[1024]; glGetShaderInfoLog(s, sizeof(log), nullptr, log);
        fprintf(stderr, "[IREngine] Shader compile hatası:\n%s\n", log);
    }
    return s;
}

static GLuint linkProgram(GLuint vert, GLuint frag) {
    GLuint p = glCreateProgram();
    glAttachShader(p, vert); glAttachShader(p, frag);
    glLinkProgram(p);
    GLint ok; glGetProgramiv(p, GL_LINK_STATUS, &ok);
    if (!ok) {
        char log[1024]; glGetProgramInfoLog(p, sizeof(log), nullptr, log);
        fprintf(stderr, "[IREngine] Program link hatası:\n%s\n", log);
    }
    glDeleteShader(vert); glDeleteShader(frag);
    return p;
}

// ─── Math Helpers ────────────────────────────────────────────────────────────
static void buildPerspective(float fovRad, float aspect, float n, float f, float out[16]) {
    for(int i=0; i<16; i++) out[i] = 0.0f;
    float tanHalfFov = tanf(fovRad / 2.0f);
    out[0] = 1.0f / (aspect * tanHalfFov);
    out[5] = 1.0f / (tanHalfFov);
    out[10] = -(f + n) / (f - n);
    out[11] = -1.0f;
    out[14] = -(2.0f * f * n) / (f - n);
}

static void buildLookAt(glm_vec3 eye, glm_vec3 center, glm_vec3 up, float out[16]) {
    glm_vec3 f = { center.x - eye.x, center.y - eye.y, center.z - eye.z };
    float flen = sqrtf(f.x*f.x + f.y*f.y + f.z*f.z);
    f.x /= flen; f.y /= flen; f.z /= flen;

    glm_vec3 s = { f.y*up.z - f.z*up.y, f.z*up.x - f.x*up.z, f.x*up.y - f.y*up.x };
    float slen = sqrtf(s.x*s.x + s.y*s.y + s.z*s.z);
    s.x /= slen; s.y /= slen; s.z /= slen;

    glm_vec3 u = { s.y*f.z - s.z*f.y, s.z*f.x - s.x*f.z, s.x*f.y - s.y*f.x };

    for(int i=0; i<16; i++) out[i] = 0.0f;
    out[0] = s.x; out[4] = s.y; out[8] = s.z;
    out[1] = u.x; out[5] = u.y; out[9] = u.z;
    out[2] = -f.x; out[6] = -f.y; out[10] = -f.z;
    out[15] = 1.0f;

    out[12] = -(s.x*eye.x + s.y*eye.y + s.z*eye.z);
    out[13] = -(u.x*eye.x + u.y*eye.y + u.z*eye.z);
    out[14] = f.x*eye.x + f.y*eye.y + f.z*eye.z;
}

// ─── Terrain Builder ─────────────────────────────────────────────────────────
struct TerrainVertex {
    float px, py, pz;
    float nx, ny, nz;
    uint32_t mat_id;
};

static void buildTerrain(GLuint& vao, GLuint& vbo, GLuint& ebo, int& index_count) {
    int gridW = 200, gridH = 200;
    float sizeX = 500.0f, sizeZ = 500.0f;

    auto noise = [](float x, float y) -> float {
        float n = sinf(x * 12.9898f + y * 78.233f) * 43758.5453f;
        return n - floorf(n);
    };
    auto smooth_noise = [&](float x, float y) {
        float fx = x - floorf(x), fy = y - floorf(y);
        fx = fx*fx*(3.0f-2.0f*fx); fy = fy*fy*(3.0f-2.0f*fy);
        float n0 = noise(floorf(x), floorf(y));
        float n1 = noise(floorf(x)+1.0f, floorf(y));
        float n2 = noise(floorf(x), floorf(y)+1.0f);
        float n3 = noise(floorf(x)+1.0f, floorf(y)+1.0f);
        return n0*(1-fx)*(1-fy) + n1*fx*(1-fy) + n2*(1-fx)*fy + n3*fx*fy;
    };
    auto fractal = [&](float x, float y) {
        return smooth_noise(x,y) + 0.5f*smooth_noise(x*2.0f, y*2.0f) + 0.25f*smooth_noise(x*4.0f, y*4.0f);
    };

    std::vector<TerrainVertex> verts;
    verts.reserve(gridW * gridH);
    for(int iy = 0; iy < gridH; iy++) {
        for(int ix = 0; ix < gridW; ix++) {
            float px = (ix / (float)(gridW-1) - 0.5f) * sizeX;
            float pz = (iy / (float)(gridH-1) - 0.5f) * sizeZ;
            float py = fractal(ix * 0.06f, iy * 0.06f) * 35.0f;

            // Ortadan yol geçsin
            float road_dist = fabsf(px + sinf(pz*0.02f)*20.0f);
            uint32_t mat = 3; // GRASS_DRY
            if (road_dist < 8.0f) {
                py *= 0.1f; // düzelt
                mat = 11; // ASPHALT
            } else if (py > 25.0f) {
                mat = 7; // ROCK
            } else if (py < 5.0f) {
                mat = 1; // MOIST_SOIL
            }

            verts.push_back({px, py, pz, 0, 1, 0, mat});
        }
    }

    // NORMALS
    for(int iy = 1; iy < gridH-1; iy++) {
        for(int ix = 1; ix < gridW-1; ix++) {
            float L = verts[iy*gridW + ix-1].py;
            float R = verts[iy*gridW + ix+1].py;
            float B = verts[(iy-1)*gridW + ix].py;
            float T = verts[(iy+1)*gridW + ix].py;
            glm_vec3 norm = {L - R, 2.0f * (sizeX/(gridW-1)), B - T};
            float len = sqrtf(norm.x*norm.x + norm.y*norm.y + norm.z*norm.z);
            verts[iy*gridW + ix].nx = norm.x / len;
            verts[iy*gridW + ix].ny = norm.y / len;
            verts[iy*gridW + ix].nz = norm.z / len;
        }
    }

    std::vector<uint16_t> idx;
    idx.reserve((gridW-1)*(gridH-1)*6);
    for(int iy = 0; iy < gridH-1; iy++) {
        for(int ix = 0; ix < gridW-1; ix++) {
            int i0 = iy*gridW + ix, i1 = i0 + 1, i2 = i0 + gridW, i3 = i2 + 1;
            idx.push_back(i0); idx.push_back(i2); idx.push_back(i1);
            idx.push_back(i1); idx.push_back(i2); idx.push_back(i3);
        }
    }

    glGenVertexArrays(1, &vao);
    glGenBuffers(1, &vbo);
    glGenBuffers(1, &ebo);

    glBindVertexArray(vao);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, verts.size() * sizeof(TerrainVertex), verts.data(), GL_STATIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, idx.size() * sizeof(uint16_t), idx.data(), GL_STATIC_DRAW);

    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(TerrainVertex), (void*)0);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(TerrainVertex), (void*)12);
    glEnableVertexAttribArray(2);
    glVertexAttribIPointer(2, 1, GL_UNSIGNED_INT, sizeof(TerrainVertex), (void*)24); // IPointer for uint!

    glBindVertexArray(0);
    index_count = idx.size();
}

// ─── Fullscreen quad ─────────────────────────────────────────────────────────
static const float kQuadVerts[] = {
    // pos       // uv
    -1.0f, -1.0f,  0.0f, 0.0f,
     1.0f, -1.0f,  1.0f, 0.0f,
     1.0f,  1.0f,  1.0f, 1.0f,
    -1.0f,  1.0f,  0.0f, 1.0f,
};
static const uint16_t kQuadIdx[] = { 0, 1, 2, 0, 2, 3 };

// ─── IREngine::init ───────────────────────────────────────────────────────────
bool IREngine::init(const Config& cfg) {
    width_  = cfg.width;
    height_ = cfg.height;
    band_flags_ = cfg.band_flags;
    int N   = width_ * height_;

    thermal_update_interval_ = (cfg.thermal_update_hz > 0) ?
        (60 / cfg.thermal_update_hz) : 5;

    // ── Materyal DB yükle ─────────────────────────────────────────────────────
    if (!cfg.material_db_path.empty()) {
        if (!materialDB_.loadFromYAML(cfg.material_db_path)) {
            fprintf(stderr, "[IREngine] Materyal DB yüklenemedi\n");
        }
        auto gpu_arr = materialDB_.buildGPUArray();
        uploadMaterialTableToGPU(
            reinterpret_cast<const GPUMaterialEntry*>(gpu_arr.data()),
            (int)gpu_arr.size()
        );
    }

    // ── Atmosfer LUT (opsiyonel) ──────────────────────────────────────────────
    if (!cfg.atm_lut_path.empty()) {
        atm_load_lut(cfg.atm_lut_path.c_str());
    }

    // ── Sensör varsayılan parametreleri ───────────────────────────────────────
    sensor_[SensorParams::SWIR] = makeSWIR();
    sensor_[SensorParams::MWIR] = makeMWIR();
    sensor_[SensorParams::LWIR] = makeLWIR();

    // CUDA bellek tahsisi
    cudaMalloc(&d_T_surface_, N * sizeof(float));
    for (int b = 0; b < 3; b++) {
        if (!(band_flags_ & (1 << b))) continue;
        cudaMalloc(&d_radiance_[b],      N * sizeof(float));
        cudaMalloc(&d_sensor_out_[b],    N * sizeof(uint16_t));
        cudaMalloc(&d_fpn_map_[b],       N * sizeof(float));
        cudaMalloc(&d_float_display_[b], N * sizeof(float));
        cudaMemset(d_float_display_[b], 0, N * sizeof(float));
    }

    // curand başlat
    sensor_init_curand((curandState**)&d_rand_states_, 42ULL, N);
    for (int b = 0; b < 3; b++) {
        if (!(band_flags_ & (1 << b))) continue;
        sensor_generate_fpn(&d_fpn_map_[b],
                             (curandState*)d_rand_states_, N);
    }

    // ── OpenGL kaynakları ─────────────────────────────────────────────────────
    // G-Buffer FBO
    glGenFramebuffers(1, &fbo_gbuffer_);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo_gbuffer_);

    auto makeTexF = [&](GLuint& tex, GLenum fmt) {
        glGenTextures(1, &tex);
        glBindTexture(GL_TEXTURE_2D, tex);
        glTexImage2D(GL_TEXTURE_2D, 0, fmt, width_, height_,
                     0, GL_RGBA, GL_FLOAT, nullptr);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    };
    makeTexF(tex_pos_,  GL_RGBA32F);
    makeTexF(tex_norm_, GL_RGBA32F);

    // Material ID — unsigned integer
    glGenTextures(1, &tex_mat_);
    glBindTexture(GL_TEXTURE_2D, tex_mat_);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32UI, width_, height_,
                 0, GL_RGBA_INTEGER, GL_UNSIGNED_INT, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex_pos_,  0);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, GL_TEXTURE_2D, tex_norm_, 0);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT2, GL_TEXTURE_2D, tex_mat_,  0);

    GLuint rbo_depth;
    glGenRenderbuffers(1, &rbo_depth);
    glBindRenderbuffer(GL_RENDERBUFFER, rbo_depth);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, width_, height_);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, rbo_depth);

    GLenum attachments[3] = {GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1, GL_COLOR_ATTACHMENT2};
    glDrawBuffers(3, attachments);

    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
        fprintf(stderr, "[IREngine] G-Buffer FBO incomplete!\n");
        return false;
    }
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    // Output texture (CUDA interop yok — glTexSubImage2D ile güncellenecek)
    for (int b = 0; b < 3; b++) {
        if (!(band_flags_ & (1 << b))) continue;
        glGenTextures(1, &tex_output_[b]);
        glBindTexture(GL_TEXTURE_2D, tex_output_[b]);
        // Sıfır initialized float texture
        std::vector<float> zeros(width_ * height_, 0.0f);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, width_, height_,
                     0, GL_RED, GL_FLOAT, zeros.data());
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        fprintf(stderr, "[IREngine] tex_output_[%d] = %u\n", b, tex_output_[b]);
    }
    glBindTexture(GL_TEXTURE_2D, 0);

    // Fullscreen quad VAO
    glGenVertexArrays(1, &vao_quad_);
    glGenBuffers(1, &vbo_quad_);
    glBindVertexArray(vao_quad_);
    glBindBuffer(GL_ARRAY_BUFFER, vbo_quad_);
    glBufferData(GL_ARRAY_BUFFER, sizeof(kQuadVerts), kQuadVerts, GL_STATIC_DRAW);

    GLuint ebo_quad;
    glGenBuffers(1, &ebo_quad);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo_quad);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(kQuadIdx), kQuadIdx, GL_STATIC_DRAW);

    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)(2*sizeof(float)));
    glBindVertexArray(0);

    // Shader programları
    auto loadProg = [&](const std::string& vp, const std::string& fp) -> GLuint {
        auto vs = compileShader(GL_VERTEX_SHADER,   readFile(vp));
        auto fs = compileShader(GL_FRAGMENT_SHADER, readFile(fp));
        return linkProgram(vs, fs);
    };

    prog_gbuf_    = loadProg("shaders/gbuffer.vert",    "shaders/gbuffer.frag");
    prog_display_ = loadProg("shaders/ir_display.vert", "shaders/ir_display.frag");

    buildTerrain(vao_terrain_, vbo_terrain_, ebo_terrain_, terrain_index_count_);

    initialized_ = true;
    printf("[IREngine] Başlatıldı: %dx%d, band_flags=0x%x\n",
           width_, height_, band_flags_);
    return true;
}

// ─── IREngine::setSensorParams ────────────────────────────────────────────────
void IREngine::setSensorParams(SensorParams::Band band, const SensorParams& p) {
    sensor_[band] = p;
}

// ─── IREngine::updateGBuffer ──────────────────────────────────────────────────
void IREngine::updateGBuffer(GLuint pos, GLuint norm, GLuint mat_id) {
    tex_pos_  = pos;
    tex_norm_ = norm;
    tex_mat_  = mat_id;
}

// ─── IREngine::renderFrame ────────────────────────────────────────────────────
void IREngine::renderFrame(const FrameParams& p) {
    if (!initialized_) return;

    // 0. G-Buffer Pass (3D Geometri -> Texture)
    glBindFramebuffer(GL_FRAMEBUFFER, fbo_gbuffer_);
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LESS);

    // Temizle: arka plan uzağı temsil eder
    GLfloat clearColor[] = {0.0f, 0.0f, 0.0f, 0.0f};
    glClearBufferfv(GL_COLOR, 0, clearColor);
    glClearBufferfv(GL_COLOR, 1, clearColor);
    GLuint clearMat[] = {26, 0, 0, 0}; // SKY_CLEAR mat id = 26
    glClearBufferuiv(GL_COLOR, 2, clearMat);
    glClear(GL_DEPTH_BUFFER_BIT);

    glUseProgram(prog_gbuf_);

    // Kamerayı hareket ettir (Z ekseninde ileriye doğru yavaşça)
    float t = frame_count_ * 0.05f; 
    glm_vec3 eye    = { sinf(t*0.2f)*20.0f, 60.0f, t*5.0f - 100.0f }; // 60m yükseklikte drone
    glm_vec3 center = { eye.x, eye.y - 15.0f, eye.z + 50.0f }; // hafif aşağı bakıyor
    glm_vec3 up     = { 0.0f, 1.0f, 0.0f };

    float viewMat[16], projMat[16];
    buildLookAt(eye, center, up, viewMat);
    buildPerspective(1.0f, (float)width_ / (float)height_, 1.0f, 600.0f, projMat);

    glUniformMatrix4fv(glGetUniformLocation(prog_gbuf_, "u_view"), 1, GL_FALSE, viewMat);
    glUniformMatrix4fv(glGetUniformLocation(prog_gbuf_, "u_proj"), 1, GL_FALSE, projMat);

    glBindVertexArray(vao_terrain_);
    glDrawElements(GL_TRIANGLES, terrain_index_count_, GL_UNSIGNED_SHORT, 0);
    glBindVertexArray(0);

    glDisable(GL_DEPTH_TEST);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    // 1. Termal solver (her N frame'de bir)
    if (frame_count_ % thermal_update_interval_ == 0) {
        stepThermalSolver(p.sim_time_hours, p.T_air_K,
                          p.wind_speed_ms, p.sun_direction);
    }

    // 2. Radyans hesabı
    computeRadiance(p);

    // 3. Sensör modeli
    applySensorModel((SensorParams::Band)p.active_band);

    // 4. CUDA → OpenGL texture transferi + display
    displayBand((SensorParams::Band)p.active_band);

    frame_count_++;
}

// ─── Dahili: Termal solver ────────────────────────────────────────────────────
void IREngine::stepThermalSolver(float sim_time_hours, float T_air_K,
                                  float wind_speed_ms, const glm_vec3& sun_dir) {
    // TODO: G-Buffer'dan normal ve material ID al, CUDA kernel çalıştır
    if (frame_count_ == 0) {
        int N = width_ * height_;
        std::vector<float> T_init(N, T_air_K);
        cudaMemcpy(d_T_surface_, T_init.data(), N * sizeof(float), cudaMemcpyHostToDevice);
    }
}

// ─── Dahili: Radyans hesabı ───────────────────────────────────────────────────
void IREngine::computeRadiance(const FrameParams& p) {
    static const float band_lam_min[] = {1.0f, 3.5f, 8.5f};
    static const float band_lam_max[] = {1.7f, 4.9f, 11.5f};

    float sun_el = (float)(M_PI * 0.25);
    float E_sol  = 1000.0f * sinf(sun_el);
    float cos_sun = sinf(sun_el);

    int N = width_ * height_;

    std::vector<float> norm_data(N * 4);
    std::vector<uint32_t> mat_data(N * 4);

    glBindTexture(GL_TEXTURE_2D, tex_norm_);
    glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_FLOAT, norm_data.data());

    glBindTexture(GL_TEXTURE_2D, tex_mat_);
    glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA_INTEGER, GL_UNSIGNED_INT, mat_data.data());
    glBindTexture(GL_TEXTURE_2D, 0);

    float t_solar = p.sim_time_hours - 12.0f;
    float cos_za  = cosf(t_solar * (float)M_PI / 12.0f);
    if (cos_za < 0.0f) cos_za = 0.0f;

    float sx = p.sun_direction.x, sy = p.sun_direction.y, sz = p.sun_direction.z;
    float wind_cool = 1.0f / (1.0f + 0.05f * p.wind_speed_ms);

    std::vector<float> T_surface(N);
    std::vector<int> M_final(N);

    for (int i = 0; i < N; i++) {
        float nx = norm_data[i*4 + 0];
        float ny = norm_data[i*4 + 1];
        float nz = norm_data[i*4 + 2];
        uint32_t m = mat_data[i*4 + 0];
        
        M_final[i] = m;
        
        if (m == 26) { // SKY
            T_surface[i] = p.T_air_K - 30.0f + (cos_za * 10.0f);
        } else {
            float dot_sun = nx*sx + ny*sy + nz*sz;
            if (dot_sun < 0.0f) dot_sun = 0.0f;
            
            float solar_factor = dot_sun * cos_za; 
            
            float mat_heat = 10.0f;
            if (m == 11) mat_heat = 20.0f; // Asfalt
            else if (m == 7) mat_heat = 15.0f; // Kaya
            else if (m == 1) mat_heat = 5.0f;  // Toprak
            
            T_surface[i] = p.T_air_K + mat_heat * solar_factor * wind_cool;
        }
    }

    cudaMemcpy(d_T_surface_, T_surface.data(), N * sizeof(float), cudaMemcpyHostToDevice);

    static int* d_mat_id_demo = nullptr;
    if (!d_mat_id_demo) {
        cudaMalloc(&d_mat_id_demo, N * sizeof(int));
    }
    cudaMemcpy(d_mat_id_demo, M_final.data(), N * sizeof(int), cudaMemcpyHostToDevice);

    // ── Her bant için Planck kernel ────────────────────────────────────────
    for (int b = 0; b < 3; b++) {
        if (!(band_flags_ & (1 << b))) continue;

        float range = p.range_km;
        float alpha = (b == 0) ? 0.10f : (b == 1) ? 0.15f : 0.05f;
        
        // Nem artışı geçirgenliği azaltır (basit yaklaşım)
        float humidity_factor = 1.0f + (p.humidity_pct - 50.0f) * 0.005f;
        float tau   = expf(-alpha * humidity_factor * range);
        float L_bb_atm = 1e-4f * expf((p.T_air_K - 288.0f) / 30.0f);
        float L_path   = (1.0f - tau) * L_bb_atm * range;
        float L_sky    = 2.0f * L_bb_atm;

        struct { int band; float lam_min_um, lam_max_um, lam_center_um; } bp{
            b, band_lam_min[b], band_lam_max[b],
            (band_lam_min[b] + band_lam_max[b]) * 0.5f
        };

        launchPlanckKernel(
            d_T_surface_, d_mat_id_demo, d_radiance_[b],
            *reinterpret_cast<BandParams*>(&bp),
            p.T_air_K, tau, L_path, L_sky, E_sol, cos_sun,
            height_, width_
        );
    }
}

// ─── Dahili: Sensör modeli / AGC / Display normalizasyon ─────────────────────
void IREngine::applySensorModel(SensorParams::Band band) {
    if (!(band_flags_ & (1 << band))) return;

    int N = width_ * height_;

    // ── Radyans buffer'ı CPU'ya al ──────────────────────────────────────────
    std::vector<float> rad(N);
    cudaMemcpy(rad.data(), d_radiance_[band], N * sizeof(float),
               cudaMemcpyDeviceToHost);

    // ── Percentile AGC: %2 - %98 arası normalize ────────────────────────────
    std::vector<float> sorted(rad);
    std::sort(sorted.begin(), sorted.end());
    int lo_idx = std::max(0, (int)(N * 0.02f));
    int hi_idx = std::min(N - 1, (int)(N * 0.98f));
    float gmin = sorted[lo_idx];
    float gmax = sorted[hi_idx];
    float range = gmax - gmin;
    if (range < 1e-12f) {
        // Tüm sahne uniform — gain/level merkezi göster
        gmin = sorted[0] * 0.99f;
        gmax = sorted[N-1] * 1.01f + 1e-10f;
        range = gmax - gmin;
    }

    // ── Gain & Level uygula, [0,1] normalize ────────────────────────────────
    // level: görüntü merkezi, gain: kontrast katsayısı
    std::vector<float> f32(N);
    for (int i = 0; i < N; i++) {
        float v = (rad[i] - gmin) / range;          // [0,1]
        v = (v - level_) * gain_ + 0.5f;            // gain/level
        f32[i] = std::max(0.0f, std::min(1.0f, v));
    }

    // ── d_float_display_ GPU buffer'a ve GL texture'a yaz ──────────────────
    cudaMemcpy(d_float_display_[band], f32.data(), N * sizeof(float),
               cudaMemcpyHostToDevice);

    // Doğrudan GL texture güncellemesi (CUDA interop yok)
    glBindTexture(GL_TEXTURE_2D, tex_output_[band]);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width_, height_,
                    GL_RED, GL_FLOAT, f32.data());
    glBindTexture(GL_TEXTURE_2D, 0);

    // Debug: ilk frame radyans aralığını yazdır
    static bool printed[3] = {};
    if (!printed[band]) {
        printed[band] = true;
        float rmin = *std::min_element(f32.begin(), f32.end());
        float rmax = *std::max_element(f32.begin(), f32.end());
        fprintf(stderr, "[IREngine] band=%d f32 range=[%.4f, %.4f]\n", band, rmin, rmax);
    }
}

// ─── Dahili: Display ──────────────────────────────────────────────────────────
void IREngine::displayBand(SensorParams::Band band) {
    if (!(band_flags_ & (1 << band))) return;
    if (!tex_output_[band]) return;

    // Display pass — texture applySensorModel içinde güncellendi
    glDisable(GL_DEPTH_TEST);
    glUseProgram(prog_display_);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, tex_output_[band]);
    glUniform1i(glGetUniformLocation(prog_display_, "u_ir_texture"), 0);
    glUniform1i(glGetUniformLocation(prog_display_, "u_colormap"),  (int)colormap_);
    glUniform1f(glGetUniformLocation(prog_display_, "u_gain"),  1.0f);   // AGC CPU-side uygulandı
    glUniform1f(glGetUniformLocation(prog_display_, "u_level"), 0.5f);   // merkez
    glUniform1i(glGetUniformLocation(prog_display_, "u_show_crosshair"),   1);
    glUniform1i(glGetUniformLocation(prog_display_, "u_show_temp_scale"),  0);
    glUniform2f(glGetUniformLocation(prog_display_, "u_resolution"),
                (float)width_, (float)height_);

    glBindVertexArray(vao_quad_);
    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_SHORT, 0);
    glBindVertexArray(0);
    glEnable(GL_DEPTH_TEST);
}

// ─── Dışa aktarım ────────────────────────────────────────────────────────────
GLuint IREngine::getOutputTexture(SensorParams::Band band) const {
    return tex_output_[band];
}

void IREngine::getRadianceBuffer(SensorParams::Band band, float* out_host) const {
    int N = width_ * height_;
    cudaMemcpy(out_host, d_radiance_[band], N * sizeof(float),
               cudaMemcpyDeviceToHost);
}

// ─── Kapatma ─────────────────────────────────────────────────────────────────
void IREngine::shutdown() {
    if (!initialized_) return;
    cudaFree(d_T_surface_);
    for (int b = 0; b < 3; b++) {
        cudaFree(d_radiance_[b]);
        cudaFree(d_sensor_out_[b]);
        cudaFree(d_fpn_map_[b]);
        if (cuda_tex_resource_[b])
            cudaGraphicsUnregisterResource(
                (cudaGraphicsResource*)cuda_tex_resource_[b]);
    }
    cudaFree(d_rand_states_);
    atm_free_lut();

    if (prog_gbuf_)    glDeleteProgram(prog_gbuf_);
    if (prog_display_) glDeleteProgram(prog_display_);
    if (vao_quad_)     glDeleteVertexArrays(1, &vao_quad_);
    if (fbo_gbuffer_)  glDeleteFramebuffers(1, &fbo_gbuffer_);

    initialized_ = false;
    printf("[IREngine] Kapatıldı.\n");
}
