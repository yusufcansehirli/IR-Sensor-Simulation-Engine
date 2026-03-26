// material_db.cpp
// ─────────────────────────────────────────────────────────────────────────────
// MaterialDB — YAML tabanlı materyal & emisivite veritabanı.
// Mimari belge §3 ile örtüşür.
// ─────────────────────────────────────────────────────────────────────────────
#include "material_db.h"
#include <yaml-cpp/yaml.h>
#include <cstdio>
#include <stdexcept>

bool MaterialDB::loadFromYAML(const std::string& path) {
    try {
        YAML::Node root = YAML::LoadFile(path);
        if (!root["materials"]) {
            fprintf(stderr, "[MaterialDB] 'materials' anahtarı bulunamadı: %s\n",
                    path.c_str());
            return false;
        }

        materials_.clear();
        nameToID_.clear();

        const YAML::Node& mats = root["materials"];
        int id = 0;
        for (auto it = mats.begin(); it != mats.end(); ++it, ++id) {
            std::string name      = it->first.as<std::string>();
            const YAML::Node& m   = it->second;

            MaterialSpectrum ms{};
            ms.name = name;

            // Bant-ortalama emisivite değerleri
            ms.swir_epsilon       = m["swir_epsilon"]       ? m["swir_epsilon"].as<float>()       : 0.90f;
            ms.mwir_epsilon       = m["mwir_epsilon"]       ? m["mwir_epsilon"].as<float>()       : 0.90f;
            ms.lwir_epsilon       = m["lwir_epsilon"]       ? m["lwir_epsilon"].as<float>()       : 0.95f;
            ms.thermal_mass       = m["thermal_mass"]       ? m["thermal_mass"].as<float>()       : 1.5e6f;
            ms.solar_absorptivity = m["solar_absorptivity"] ? m["solar_absorptivity"].as<float>() : 0.75f;
            ms.temperature_offset_K = m["temperature_offset_K"] ?
                                        m["temperature_offset_K"].as<float>() : 0.0f;

            // Sürekli spektrum (opsiyonel)
            if (m["wavelengths"] && m["emissivities"]) {
                auto wl = m["wavelengths"].as<std::vector<float>>();
                auto em = m["emissivities"].as<std::vector<float>>();
                if (wl.size() == em.size()) {
                    ms.wavelength_um = wl;
                    ms.emissivity    = em;
                }
            }

            materials_.push_back(std::move(ms));
            nameToID_[name] = id;
        }

        printf("[MaterialDB] %zu materyal yüklendi: %s\n",
               materials_.size(), path.c_str());
        return true;

    } catch (const YAML::Exception& e) {
        fprintf(stderr, "[MaterialDB] YAML parse hatası: %s\n", e.what());
        return false;
    }
}

const MaterialSpectrum* MaterialDB::getByID(int id) const {
    if (id < 0 || id >= (int)materials_.size()) return nullptr;
    return &materials_[id];
}

const MaterialSpectrum* MaterialDB::getByName(const std::string& name) const {
    auto it = nameToID_.find(name);
    if (it == nameToID_.end()) return nullptr;
    return &materials_[it->second];
}

std::vector<MaterialDB::GPUMaterialEntry> MaterialDB::buildGPUArray() const {
    std::vector<GPUMaterialEntry> arr(materials_.size());
    for (size_t i = 0; i < materials_.size(); i++) {
        const auto& m = materials_[i];
        arr[i].swir_eps            = m.swir_epsilon;
        arr[i].mwir_eps            = m.mwir_epsilon;
        arr[i].lwir_eps            = m.lwir_epsilon;
        arr[i].thermal_mass        = m.thermal_mass;
        arr[i].solar_absorptivity  = m.solar_absorptivity;
        arr[i].temperature_offset_K= m.temperature_offset_K;
        arr[i]._pad[0] = arr[i]._pad[1] = 0.0f;
    }
    return arr;
}
