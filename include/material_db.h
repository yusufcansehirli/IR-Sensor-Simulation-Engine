#pragma once
#include <string>
#include <vector>
#include <unordered_map>

// ─────────────────────────────────────────────────────────────────────────────
// material_db.h  —  Materyal & Emisivite Veritabanı
// Mimari belge §3 ile birebir örtüşür.
// ─────────────────────────────────────────────────────────────────────────────

struct MaterialSpectrum {
    std::string name;
    std::vector<float> wavelength_um;   // μm cinsinden, artan sırada
    std::vector<float> emissivity;      // 0.0 – 1.0
    float thermal_mass;                 // J/(m³·K)
    float solar_absorptivity;           // 0.0 – 1.0
    float temperature_offset_K;         // Gövde sıcaklığına ek (motor vb.)

    // Bant-ortalama önbellekleri (YAML'dan doğrudan okunur)
    float swir_epsilon;
    float mwir_epsilon;
    float lwir_epsilon;
};

// ─── Lineer interpolasyon ile emisivite ────────────────────────────────────────
inline float getEmissivity(const MaterialSpectrum& mat, float lambda_um) {
    if (mat.wavelength_um.empty()) {
        // Sürekli spektrum yok — bant sabitlerini seç
        if (lambda_um < 2.5f) return mat.swir_epsilon;
        if (lambda_um < 5.0f) return mat.mwir_epsilon;
        return mat.lwir_epsilon;
    }
    // Lineer interpolasyon
    auto it = std::lower_bound(mat.wavelength_um.begin(),
                               mat.wavelength_um.end(), lambda_um);
    if (it == mat.wavelength_um.end())   return mat.emissivity.back();
    if (it == mat.wavelength_um.begin()) return mat.emissivity.front();
    size_t idx = static_cast<size_t>(it - mat.wavelength_um.begin());
    float t = (lambda_um - mat.wavelength_um[idx - 1]) /
              (mat.wavelength_um[idx] - mat.wavelength_um[idx - 1]);
    return mat.emissivity[idx - 1] + t * (mat.emissivity[idx] - mat.emissivity[idx - 1]);
}

// ─── Veritabanı sınıfı ────────────────────────────────────────────────────────
class MaterialDB {
public:
    // YAML dosyasından yükle (yaml-cpp)
    bool loadFromYAML(const std::string& path);

    // ID ile erişim (integer — G-Buffer materyel ID'si)
    const MaterialSpectrum* getByID(int id) const;

    // İsim ile erişim
    const MaterialSpectrum* getByName(const std::string& name) const;

    int count() const { return static_cast<int>(materials_.size()); }

    // GPU'ya yüklemek için düz dizi yardımcısı (bant-ortalama değerler)
    struct GPUMaterialEntry {
        float swir_eps, mwir_eps, lwir_eps;
        float thermal_mass;
        float solar_absorptivity;
        float temperature_offset_K;
        float _pad[2];
    };

    // materials[id] → GPUMaterialEntry dizisini doldurur
    std::vector<GPUMaterialEntry> buildGPUArray() const;

private:
    std::vector<MaterialSpectrum>                materials_;
    std::unordered_map<std::string, int>         nameToID_;
};
