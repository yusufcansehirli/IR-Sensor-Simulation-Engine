// main.cpp
// ─────────────────────────────────────────────────────────────────────────────
// IR Sensör Simülatörü — Standalone test uygulaması.
// GLFW penceresi + ImGui kontrol paneli.
// Mimari belge §6.5 ile örtüşür.
// ─────────────────────────────────────────────────────────────────────────────
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <imgui.h>
#include <backends/imgui_impl_glfw.h>
#include <backends/imgui_impl_opengl3.h>

#include <cstdio>
#include <cmath>
#include <string>

#include "ir_engine.h"
#include "sensor_params.h"

// ─── Uygulama durumu ────────────────────────────────────────────────────────
static int   WIN_W  = 1280;
static int   WIN_H  = 800;
static int   RENDER_W = 640;
static int   RENDER_H = 512;

struct AppState {
    // Sim parametreleri
    float sim_time_hours   = 12.0f;     // Öğle saati başlangıç
    float range_km         = 2.0f;
    float T_air_K          = 293.0f;    // 20°C
    float humidity_pct     = 50.0f;
    float wind_speed_ms    = 3.0f;

    // Sensör seçimi
    int   active_band      = 1;         // MWIR
    int   colormap         = 2;         // Whitehot
    float gain             = 1.5f;
    float level            = 0.5f;

    // Simülasyon kontrol
    bool  running          = true;
    bool  realtime_diurnal = false;     // Gerçek zamanlı gün döngüsü

    // Gürültü parametreleri (anlık override)
    float nedt_mk          = 25.0f;
    float fpn_fraction     = 0.02f;
};

// ─── GLFW hata callback ─────────────────────────────────────────────────────
static void glfwErrorCallback(int code, const char* msg) {
    fprintf(stderr, "[GLFW] Hata %d: %s\n", code, msg);
}

// ─── ImGui panel çizimi ─────────────────────────────────────────────────────
static void drawControlPanel(AppState& state, IREngine& engine) {
    ImGui::SetNextWindowPos(ImVec2(RENDER_W + 10, 10), ImGuiCond_Once);
    ImGui::SetNextWindowSize(ImVec2(300, WIN_H - 20), ImGuiCond_Once);
    ImGui::Begin("IR Sim Kontrolü", nullptr,
                 ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoResize);

    // ── Sahne Parametreleri ──────────────────────────────────────────────────
    ImGui::SeparatorText("Sahne / Atmosfer");

    ImGui::SliderFloat("Saat", &state.sim_time_hours, 0.0f, 24.0f, "%.1f h");
    ImGui::SliderFloat("Mesafe [km]", &state.range_km, 0.1f, 10.0f, "%.2f");
    ImGui::SliderFloat("Hava Sıcaklığı [K]", &state.T_air_K, 250.0f, 330.0f, "%.1f");
    ImGui::SliderFloat("Nem [%]", &state.humidity_pct, 10.0f, 95.0f, "%.0f");
    ImGui::SliderFloat("Rüzgar [m/s]", &state.wind_speed_ms, 0.0f, 20.0f, "%.1f");
    ImGui::Checkbox("Gerçek Zamanlı Diurnal", &state.realtime_diurnal);

    // ── Bant & Görüntü ───────────────────────────────────────────────────────
    ImGui::SeparatorText("Bant & Görüntü");
    {
        const char* bands[] = {"SWIR", "MWIR", "LWIR"};
        ImGui::Combo("Aktif Bant", &state.active_band, bands, 3);
    }
    {
        const char* cmaps[] = {"Grayscale", "Blackhot", "Whitehot", "Rainbow", "Iron"};
        if (ImGui::Combo("Colormap", &state.colormap, cmaps, 5)) {
            engine.setColormap((IREngine::Colormap)state.colormap);
        }
    }
    if (ImGui::SliderFloat("Gain", &state.gain, 0.1f, 5.0f)) {
        engine.setGainLevel(state.gain, state.level);
    }
    if (ImGui::SliderFloat("Level", &state.level, 0.0f, 1.0f)) {
        engine.setGainLevel(state.gain, state.level);
    }

    // ── Sensör Gürültü ───────────────────────────────────────────────────────
    ImGui::SeparatorText("Sensör Gürültü");
    bool noise_changed = false;
    noise_changed |= ImGui::SliderFloat("NEdT [mK]",   &state.nedt_mk,    5.0f, 200.0f);
    noise_changed |= ImGui::SliderFloat("FPN sigma",   &state.fpn_fraction,0.0f, 0.1f, "%.3f");
    if (noise_changed) {
        SensorParams sp = (state.active_band == 0) ? makeSWIR() :
                          (state.active_band == 1) ? makeMWIR() : makeLWIR();
        sp.nedt_mk            = state.nedt_mk;
        sp.fpn_sigma_fraction = state.fpn_fraction;
        engine.setSensorParams((SensorParams::Band)state.active_band, sp);
    }

    // ── Kontrol ──────────────────────────────────────────────────────────────
    ImGui::SeparatorText("Kontrol");
    if (ImGui::Button(state.running ? "Durdur" : "Devam")) {
        state.running = !state.running;
    }
    ImGui::SameLine();
    if (ImGui::Button("Sıfırla")) {
        state.sim_time_hours = 12.0f;
        state.T_air_K = 293.0f;
        engine.setColormap(IREngine::WHITEHOT);
        engine.setGainLevel(1.5f, 0.5f);
    }

    // ── Durum bilgisi ────────────────────────────────────────────────────────
    ImGui::SeparatorText("Durum");
    ImGui::Text("FPS: %.1f", ImGui::GetIO().Framerate);
    float t_celsius = state.T_air_K - 273.15f;
    ImGui::Text("T_air: %.1f°C / %.1f K", t_celsius, state.T_air_K);
    float t_solar_deg = fabsf(state.sim_time_hours - 12.0f) * 15.0f;
    ImGui::Text("Solar zenith: %.0f°", t_solar_deg);
    const char* band_names[] = {"SWIR", "MWIR", "LWIR"};
    ImGui::Text("Bant: %s | Colormap: %d", band_names[state.active_band], state.colormap);

    // ── Klavye kısayolları ───────────────────────────────────────────────────
    ImGui::SeparatorText("Kısayollar");
    ImGui::TextDisabled("1/2/3: Bant seç");
    ImGui::TextDisabled("C: Colormap döndür");
    ImGui::TextDisabled("Space: Durdur/Devam");
    ImGui::TextDisabled("ESC: Çıkış");

    ImGui::End();
}

// ─── Main ──────────────────────────────────────────────────────────────────
int main(int argc, char* argv[]) {
    printf("=== IR Sensor Simulation Engine — Faz 1 ===\n");

    // ── GLFW ─────────────────────────────────────────────────────────────────
    glfwSetErrorCallback(glfwErrorCallback);
    if (!glfwInit()) { fprintf(stderr, "GLFW başlatılamadı\n"); return 1; }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 5);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_SAMPLES, 0);   // MSAA yok — IR görüntü için uygun değil

    std::string title = "IR Sensor Sim — Faz 1 [OpenGL + CUDA]";
    GLFWwindow* window = glfwCreateWindow(WIN_W, WIN_H, title.c_str(),
                                          nullptr, nullptr);
    if (!window) { fprintf(stderr, "GLFW pencere oluşturulamadı\n"); return 1; }
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);    // VSync

    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        fprintf(stderr, "GLAD başlatılamadı\n"); return 1;
    }
    printf("[GL] Versiyon: %s\n", glGetString(GL_VERSION));
    printf("[GL] Renderer: %s\n", glGetString(GL_RENDERER));

    // ── ImGui ─────────────────────────────────────────────────────────────────
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;

    ImGui::StyleColorsDark();
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 450");

    // ── IR Engine ─────────────────────────────────────────────────────────────
    IREngine engine;
    IREngine::Config cfg{};
    cfg.width               = RENDER_W;
    cfg.height              = RENDER_H;
    cfg.band_flags          = 0b111;   // SWIR + MWIR + LWIR
    cfg.material_db_path    = "data/material_database.yaml";
    cfg.enable_thermal_solver = true;
    cfg.thermal_update_hz   = 10;

    // Komut satırı: LUT yolu opsiyonel
    if (argc > 1) cfg.atm_lut_path = argv[1];

    if (!engine.init(cfg)) {
        fprintf(stderr, "IR Engine başlatılamadı\n");
        return 1;
    }

    AppState state;
    engine.setColormap(IREngine::WHITEHOT);
    engine.setGainLevel(state.gain, state.level);

    // ── OpenGL viewport yapılandırması ────────────────────────────────────────
    GLuint vp_fbo = 0;   // 0 → default framebuffer'a render
    glViewport(0, 0, RENDER_W, RENDER_H);
    glClearColor(0.05f, 0.05f, 0.05f, 1.0f);

    // ── Ana döngü ─────────────────────────────────────────────────────────────
    double last_time = glfwGetTime();

    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();

        double now = glfwGetTime();
        double dt  = now - last_time;
        last_time  = now;

        // ── Klavye kısayolları ────────────────────────────────────────────────
        if (!io.WantCaptureKeyboard) {
            if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
                glfwSetWindowShouldClose(window, GLFW_TRUE);
            if (glfwGetKey(window, GLFW_KEY_1) == GLFW_PRESS) state.active_band = 0;
            if (glfwGetKey(window, GLFW_KEY_2) == GLFW_PRESS) state.active_band = 1;
            if (glfwGetKey(window, GLFW_KEY_3) == GLFW_PRESS) state.active_band = 2;
            static bool c_pressed = false;
            if (glfwGetKey(window, GLFW_KEY_C) == GLFW_PRESS && !c_pressed) {
                state.colormap = (state.colormap + 1) % 5;
                engine.setColormap((IREngine::Colormap)state.colormap);
                c_pressed = true;
            }
            if (glfwGetKey(window, GLFW_KEY_C) == GLFW_RELEASE) c_pressed = false;
        }

        // ── Diurnal güncelleme ────────────────────────────────────────────────
        if (state.running && state.realtime_diurnal) {
            state.sim_time_hours += (float)(dt / 3600.0);   // gerçek zamanlı
            if (state.sim_time_hours >= 24.0f) state.sim_time_hours -= 24.0f;
        }

        // ── IR Render ─────────────────────────────────────────────────────────
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        glViewport(0, 0, RENDER_W, RENDER_H);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        if (state.running) {
            IREngine::FrameParams fp{};
            fp.sim_time_hours = state.sim_time_hours;
            fp.range_km       = state.range_km;
            fp.T_air_K        = state.T_air_K;
            fp.humidity_pct   = state.humidity_pct;
            fp.wind_speed_ms  = state.wind_speed_ms;
            fp.sun_direction  = {0.0f, 0.707f, 0.707f};
            fp.active_band    = state.active_band;
            engine.renderFrame(fp);
        }

        // ── ImGui ─────────────────────────────────────────────────────────────
        glViewport(0, 0, WIN_W, WIN_H);
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        drawControlPanel(state, engine);

        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        glfwSwapBuffers(window);
    }

    // ── Temizlik ──────────────────────────────────────────────────────────────
    engine.shutdown();
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    glfwDestroyWindow(window);
    glfwTerminate();

    printf("=== Çıkış ===\n");
    return 0;
}
