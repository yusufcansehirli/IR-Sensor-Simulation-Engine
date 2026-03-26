// tests/test_physics.cpp
// ─────────────────────────────────────────────────────────────────────────────
// IR Sensör Simülatörü — Fizik Birim Testleri.
// Catch2 gerektirmez — basit assert tabanlı.
// ─────────────────────────────────────────────────────────────────────────────
#include <cstdio>
#include <cmath>
#include <cassert>
#include <cstring>
#include <string>
#include "material_db.h"

// CUDA host yardımcı fonksiyonları (planck_kernel.cu'dan)
extern "C" {
    float planck_host(float T, float lambda_um);
    float band_radiance_host(float T, float eps, float lam_min, float lam_max, int N);
    float solar_irradiance_host(float hour);
}

// ─── Test yardımcısı ─────────────────────────────────────────────────────────
static int pass_count = 0;
static int fail_count = 0;

static void CHECK(bool cond, const std::string& name, const std::string& msg = "") {
    if (cond) {
        printf("  [PASS] %s\n", name.c_str());
        pass_count++;
    } else {
        printf("  [FAIL] %s  %s\n", name.c_str(), msg.c_str());
        fail_count++;
    }
}

static void CHECK_NEAR(float actual, float expected, float tol_frac,
                        const std::string& name) {
    float err = fabsf(actual - expected) / (fabsf(expected) + 1e-30f);
    std::string msg = "(actual=" + std::to_string(actual) +
                      " expected=" + std::to_string(expected) +
                      " err=" + std::to_string(err * 100) + "%)";
    CHECK(err < tol_frac, name, msg);
}

// ─── 1. Planck fonksiyonu testleri ──────────────────────────────────────────
void test_planck() {
    printf("\n[TEST] Planck Fonksiyonu\n");

    // T=300K, λ=10μm — planck_host [W/m²/sr/m] döndürür → 9.5 W/m²/sr/μm = 9.5e6 W/m²/sr/m
    float L = planck_host(300.0f, 10.0f);
    CHECK_NEAR(L, 9.5e6f, 0.10f,  "Planck(300K, 10μm) ≈ 9.5e6 W/m²/sr/m");

    // T=0K → 0
    float L_zero = planck_host(0.0f, 10.0f);
    CHECK(L_zero == 0.0f, "Planck(0K) = 0");

    // Sınır: negatif lambda → 0
    float L_neg = planck_host(300.0f, -1.0f);
    CHECK(L_neg == 0.0f, "Planck(T, λ<0) = 0");

    // Monotonik: T yükseldikçe L artar
    float L1 = planck_host(300.0f, 10.0f);
    float L2 = planck_host(310.0f, 10.0f);
    CHECK(L2 > L1, "Planck monotonik: T artar → L artar");

    // Wien verimi: λ_max * T ≈ 2898 μm·K
    // 300K → λ_max ≈ 9.66 μm
    float L_9_6  = planck_host(300.0f, 9.6f);
    float L_5_0  = planck_host(300.0f, 5.0f);
    float L_15_0 = planck_host(300.0f, 15.0f);
    CHECK(L_9_6 > L_5_0 && L_9_6 > L_15_0,
          "Wien: 300K için maksimum λ ≈ 9.66μm");
}

// ─── 2. Bant entegrasyonu testleri ──────────────────────────────────────────
void test_band_radiance() {
    printf("\n[TEST] Bant Entegrasyonu\n");

    // LWIR bant radyansı 300K, ε=0.95 — yaklaşık değer
    float L_lwir = band_radiance_host(300.0f, 0.95f, 8.5f, 11.5f, 51);
    printf("  LWIR L(300K, ε=0.95) = %.4e W/m²/sr\n", L_lwir);
    CHECK(L_lwir > 0.0f, "LWIR bant radyansı > 0");

    // MWIR @ 300K, ε=0.92
    float L_mwir = band_radiance_host(300.0f, 0.92f, 3.5f, 4.9f, 51);
    printf("  MWIR L(300K, ε=0.92) = %.4e W/m²/sr\n", L_mwir);
    CHECK(L_mwir > 0.0f, "MWIR bant radyansı > 0");

    // LWIR > MWIR @ 300K (emisyon dominant region)
    CHECK(L_lwir > L_mwir,
          "LWIR > MWIR @ 300K (LWIR dominant)");

    // Monotonik: T artar → L artar (LWIR)
    float L_260 = band_radiance_host(260.0f, 0.95f, 8.5f, 11.5f, 51);
    float L_350 = band_radiance_host(350.0f, 0.95f, 8.5f, 11.5f, 51);
    CHECK(L_350 > L_260,
          "Bant radyansı monotonik (LWIR)");

    // ε=0 → L=0
    float L_black = band_radiance_host(300.0f, 0.0f, 8.5f, 11.5f, 51);
    CHECK(L_black < 1e-20f, "ε=0 → L ≈ 0");

    // ε=1 en büyük
    float L_eps1 = band_radiance_host(300.0f, 1.0f, 8.5f, 11.5f, 51);
    float L_eps05= band_radiance_host(300.0f, 0.5f, 8.5f, 11.5f, 51);
    CHECK_NEAR(L_eps05, L_eps1 * 0.5f, 0.02f,
               "L(ε=0.5) ≈ L(ε=1) × 0.5");
}

// ─── 3. Solar irradyans testleri ────────────────────────────────────────────
void test_solar_irradiance() {
    printf("\n[TEST] Solar Irradyans\n");

    float E_noon  = solar_irradiance_host(12.0f);  // Öğle
    float E_dawn  = solar_irradiance_host(6.0f);   // Gündoğumu
    float E_night = solar_irradiance_host(0.0f);   // Gece

    printf("  E_noon(12h) = %.1f W/m²\n", E_noon);
    printf("  E_dawn(6h)  = %.1f W/m²\n", E_dawn);
    printf("  E_night(0h) = %.1f W/m²\n", E_night);

    CHECK(E_noon > 900.0f, "Öğle: E > 900 W/m²");
    CHECK(E_night < 10.0f, "Gece: E < 10 W/m²");
    CHECK(E_noon > E_dawn,  "Öğle > Gündoğumu");
}

// ─── 4. Materyal DB testleri ─────────────────────────────────────────────────
void test_material_db() {
    printf("\n[TEST] Materyal Veritabanı\n");

    MaterialDB db;
    bool loaded = db.loadFromYAML("data/material_database.yaml");
    CHECK(loaded, "YAML dosyası yüklendi");

    if (!loaded) return;

    CHECK(db.count() > 10, "Veritabanında > 10 materyal var");

    const auto* dry_soil = db.getByName("dry_soil");
    CHECK(dry_soil != nullptr, "dry_soil materyali bulundu");
    if (dry_soil) {
        printf("  dry_soil: LWIR ε = %.2f\n", dry_soil->lwir_epsilon);
        CHECK(dry_soil->lwir_epsilon > 0.9f && dry_soil->lwir_epsilon < 1.0f,
              "dry_soil LWIR ε ∈ (0.9, 1.0)");
        CHECK(dry_soil->thermal_mass > 1e5f, "thermal_mass > 0");
    }

    const auto* aluminum = db.getByName("aluminum_bare");
    CHECK(aluminum != nullptr, "aluminum_bare materyali bulundu");
    if (aluminum) {
        printf("  aluminum_bare: SWIR ε = %.2f\n", aluminum->swir_epsilon);
        CHECK(aluminum->swir_epsilon < 0.1f, "Alüminyum: düşük SWIR emisivite");
    }

    const auto* exhaust = db.getByName("engine_exhaust_hot");
    CHECK(exhaust != nullptr, "engine_exhaust_hot materyali bulundu");
    if (exhaust) {
        CHECK(exhaust->temperature_offset_K > 100.0f,
              "Egzoz: temperature_offset > 100K");
    }

    // GPU array dönüşümü
    auto gpu_arr = db.buildGPUArray();
    CHECK((int)gpu_arr.size() == db.count(), "GPU array boyutu = materyal sayısı");
}

// ─── 5. Beer-Lambert testi ───────────────────────────────────────────────────
void test_beer_lambert() {
    printf("\n[TEST] Beer-Lambert Yaklaşımı\n");

    // τ = exp(-α × r)
    // α_LWIR = 0.05/km, r = 1km → τ ≈ 0.951
    float alpha = 0.05f;
    float r     = 1.0f;
    float tau   = expf(-alpha * r);
    printf("  τ(LWIR, 1km) = %.4f\n", tau);
    CHECK_NEAR(tau, 0.951f, 0.01f, "LWIR τ(1km) ≈ 0.951");

    // r = 0 → τ = 1
    float tau_0 = expf(-alpha * 0.0f);
    CHECK_NEAR(tau_0, 1.0f, 0.001f, "τ(r=0) = 1");

    // r → ∞ → τ → 0
    float tau_inf = expf(-alpha * 100.0f);
    CHECK(tau_inf < 0.01f, "τ(r=100km) < 0.01");
}

// ─── Ana test fonksiyonu ─────────────────────────────────────────────────────
int main() {
    printf("╔══════════════════════════════════════════════╗\n");
    printf("║  IR Sensor Sim — Fizik Birim Testleri       ║\n");
    printf("╚══════════════════════════════════════════════╝\n");

    test_planck();
    test_band_radiance();
    test_solar_irradiance();
    test_material_db();
    test_beer_lambert();

    printf("\n─────────────────────────────────────────────\n");
    printf("Sonuç: %d PASS, %d FAIL\n", pass_count, fail_count);
    printf("─────────────────────────────────────────────\n");

    return (fail_count > 0) ? 1 : 0;
}
