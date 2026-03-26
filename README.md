<a id="readme-top"></a>

***🚧 Bu repository bir Vibe Coding projesidir. / This is a Vibe Coding project. 🚧***

> 🌍 **Languages / Diller:**
> - [🇬🇧 English Version](#english-documentation)
> - [🇹🇷 Türkçe Versiyon](#türkçe-dokümantasyon)

<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/yusufcansehirli/IR-Sensor-Simulation-Engine">
    <h1 align="center">IR Sensor Simulation Engine</h1>
  </a>

  <p align="center">
    High-performance, real-time physically based Thermal/Infrared Sensor Simulator
    <br />
    <br />
    <a href="#usage">View Demo</a>
    ·
    <a href="https://github.com/yusufcansehirli/IR-Sensor-Simulation-Engine/issues/new?labels=bug&template=bug-report---.md">Report Bug</a>
    ·
    <a href="https://github.com/yusufcansehirli/IR-Sensor-Simulation-Engine/issues/new?labels=enhancement&template=feature-request---.md">Request Feature</a>
  </p>
</div>


---

<a id="english-documentation"></a>
## 🇬🇧 English Documentation

<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
  </ol>
</details>

### About The Project

This project simulates real-world infrared physics and produces synthetic thermal imagery directly from generic 3D metrics. The system natively reads generated terrains via a G-Buffer pipeline (Material ID & Normal distributions) and feeds them directly to CUDA for hardware-accelerated radiance physics execution.

**Core Features:**
- **GPU-Accelerated Thermal Physics**: Uses CUDA kernels to perform real-time thermal balancing based on solar radiation (Lambertian principles), ambient temperature, wind convection, and terrain material heat properties.
- **Advanced Environment Modeling**: Fully dynamic diurnal cycles. The simulation organically adapts to the time of day, absolute temperature, humidity (affecting atmospheric transmission via Beer-Lambert modeling), and wind conditions.
- **Physical Sensor Simulation**: Implements MTF (Modulation Transfer Function) with separable Gaussian PSF blurring and incorporates comprehensive FPA models (`NEdT`, `FPN`), including non-uniformity spatial noise, temporal noise, and simulated AGC logic mapping radiance ranges dynamically to display limits.
- **Hybrid Target Pipelines**: Integrates OpenGL's MRT (Multiple Render Targets) to directly output Position, Normal, and Material IDs via G-Buffers.
- **Extensible Integration**: Designed as both a standalone executable (`ir_sim`) equipped with a real-time Dear ImGui dashboard, and as a native `ir_sensor_plugin.so` for easy embedment into platforms like Unity and Unreal Engine 5.

<p align="right">(<a href="#readme-top">back to top</a>)</p>


### Built With

* [![CUDA][CUDA-Shield]][CUDA-url]
* [![OpenGL][OpenGL-Shield]][OpenGL-url]
* [![C++][Cpp-Shield]][Cpp-url]
* [![ImGui][ImGui-Shield]][ImGui-url]

<p align="right">(<a href="#readme-top">back to top</a>)</p>


### Getting Started

To get a local copy up and running follow these simple example steps.

#### Prerequisites

* **CUDA Toolkit** (Compute Capability 7.5+ recommended)
* **OpenGL 4.5+ & GLFW3**
* **yaml-cpp** (For material DB loading)
* **HDF5** (C/C++ libraries for atmospheric lookup tables)

#### Installation

1. Clone the repo
   ```sh
   git clone https://github.com/yusufcansehirli/IR-Sensor-Simulation-Engine.git
   ```
2. Build the project
   ```sh
   mkdir build
   cd build
   cmake -DCMAKE_BUILD_TYPE=Release ..
   make -j16
   ```

<p align="right">(<a href="#readme-top">back to top</a>)</p>


### Usage

Run the main simulator to observe the runtime visual output with customizable GUI configurations:
```sh
./build/ir_sim
```
You can dynamically alter time of day, wind speed, ambient temperatures, and switch active imaging bands (e.g., MWIR 3.5 - 4.9 µm vs. LWIR 8.5 - 11.5 µm) to experience real-time heat adaptations.

<p align="right">(<a href="#readme-top">back to top</a>)</p>


### Roadmap

- [x] Phase 1: 3D Terrain & G-Buffer Rendering Integration
- [ ] Phase 2: Sensor Tracker Integration (FocusTrack) 
- [ ] Phase 3: Unreal Engine 5 Custom Plugin Support

See the [open issues](https://github.com/yusufcansehirli/IR-Sensor-Simulation-Engine/issues) for a full list of proposed features (and known issues).

<p align="right">(<a href="#readme-top">back to top</a>)</p>



---

<a id="türkçe-dokümantasyon"></a>
## 🇹🇷 Türkçe Dokümantasyon

<details>
  <summary>İçindekiler / Table of Contents</summary>
  <ol>
    <li>
      <a href="#proje-hakkında">Proje Hakkında</a>
      <ul>
        <li><a href="#kullanılan-teknolojiler">Kullanılan Teknolojiler</a></li>
      </ul>
    </li>
    <li>
      <a href="#başlarken">Başlarken</a>
      <ul>
        <li><a href="#gereksinimler">Gereksinimler</a></li>
        <li><a href="#kurulum">Kurulum</a></li>
      </ul>
    </li>
    <li><a href="#kullanım">Kullanım</a></li>
    <li><a href="#yol-haritası">Yol Haritası</a></li>
    <li><a href="#katkıda-bulunma">Katkıda Bulunma</a></li>
    <li><a href="#lisans">Lisans</a></li>
  </ol>
</details>

### Proje Hakkında

**IR Sensor Simulation Engine**, Bilgisayarlı Görme, Hedef Takibi ve İHA (Drone) simülasyonları için özel olarak geliştirilmiş yüksek performanslı ve gerçek zamanlı fiziksel tabanlı bir Termal/Kızılötesi Sensör Simülatörüdür. 3B ortamları işlemek, termal radyans hesaplamaları yapmak ve gerçekçi sensör hatalarını (noise) uygulamak için hibrit **OpenGL + CUDA** mimarisinden yararlanır.

**Temel Özellikler:**
- **GPU Hızlandırmalı Termal Fizik:** Güneş radyasyonu (Lambertian yasası), ortam hava sıcaklığı, rüzgar kaynaklı ısı taşınımı (konveksiyon) ve materyallerin fiziksel kapasitelerine dayalı gerçek zamanlı termal denge (T_surface) çözümleri üretir.
- **Gelişmiş Çevresel Modelleme:** Tamamen dinamik gün içi döngüler! Simülasyon, günün saatine, sıcaklığa, rüzgar durumuna ve atmosferik geçirgenliğe dinamik olarak uyum sağlar.
- **Fiziksel Sensör Simülasyonu:** Gelişmiş MTF hesaplaması ile Optik Dağılımını (PSF), Mekansal (FPN) ve Zamansal (NEdT) gürültüleri simüle eder. Gerçek boyutlu dinamik AGC haritalamasını barındırır.
- **3D G-Buffer Entegrasyonu:** OpenGL'in Çoklu Render Hedefi (MRT) mimarisi ile Pozisyon, Normal Vektörleri ve Materyal ID'leri doğrudan CUDA tabanlı ışın/enerji hesaplama fonksiyonlarına donanım seviyesinde paslanır.
- **Esnek Entegrasyon:** Gerçek zamanlı Dear ImGui kontrol paneline sahip bağımsız çalıştırılabilir (`ir_sim`) versiyonunun yanı sıra, simülasyon motorlarına kolayca bağlanabilen `ir_sensor_plugin.so` kütüphane çıktısını destekler.

<p align="right">(<a href="#readme-top">başa dön</a>)</p>


### Kullanılan Teknolojiler

* [![CUDA][CUDA-Shield]][CUDA-url]
* [![OpenGL][OpenGL-Shield]][OpenGL-url]
* [![C++][Cpp-Shield]][Cpp-url]
* [![ImGui][ImGui-Shield]][ImGui-url]

<p align="right">(<a href="#readme-top">başa dön</a>)</p>


### Başlarken

Projeyi yerel makinenizde çalıştırmak için aşağıdaki adımları izleyin.

#### Gereksinimler

* **CUDA Toolkit** (Compute Capability 7.5+ önerilir)
* **OpenGL 4.5+ & GLFW3**
* **yaml-cpp** (Materyal Veritabanı okumaları için)
* **HDF5** (Atmosferik kütüphane geçişleri için)

#### Kurulum

1. Depoyu klonlayın
   ```sh
   git clone https://github.com/yusufcansehirli/IR-Sensor-Simulation-Engine.git
   ```
2. Projeyi CMake ile derleyin
   ```sh
   mkdir build
   cd build
   cmake -DCMAKE_BUILD_TYPE=Release ..
   make -j16
   ```

<p align="right">(<a href="#readme-top">başa dön</a>)</p>


### Kullanım

Görsel çıktı değerlerini anlık izlemek ve sahne ile etkileşime geçmek için temel simülatörü çalıştırın:
```sh
./build/ir_sim
```
GUI üzerinden simülasyon saati (`sim_time_hours`), rüzgar, nem ile kamera bandı algılayıcısını (MWIR veya LWIR) anlık olarak değiştirebilir ve sahnedeki termal tepkimelerin gerçek hayattaki gibi anlık değiştiğini gözlemleyebilirsiniz.

<p align="right">(<a href="#readme-top">başa dön</a>)</p>


### Yol Haritası

- [x] Faz 1: 3D Geometri ve G-Buffer Rendering Entegrasyonu
- [ ] Faz 2: Sensör İzleyici/Tracker Entegrasyonu (FocusTrack)
- [ ] Faz 3: Unreal Engine 5 Custom Plugin Bağlantıları (ir_sensor_plugin)

Önerilen tüm projeleri / hataları görmek için [open issues](https://github.com/yusufcansehirli/IR-Sensor-Simulation-Engine/issues) sayfasını ziyaret edebilirsiniz.

<p align="right">(<a href="#readme-top">başa dön</a>)</p>

### Katkıda Bulunma

Açık kaynak komünitesine her türlü katkı çok değerlidir. Bu yönergeler doğrultusunda Pull Request açmaktan çekinmeyin!

<p align="right">(<a href="#readme-top">başa dön</a>)</p>

### License

Distributed under the MIT License. See `LICENSE.txt` for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Lisans

Proje MIT Lisansı ile dağıtılmaktadır. Daha fazla bilgi için `LICENSE.txt` dosyasına bakabilirsiniz.

<p align="right">(<a href="#readme-top">başa dön</a>)</p>

<a id="contact"></a>
### Contact & İletişim

Yusufcan Şehirli - [yusufcansehirli@gmail.com](mailto:yusufcansehirli@gmail.com)

Project Link: [https://github.com/yusufcansehirli/IR-Sensor-Simulation-Engine](https://github.com/yusufcansehirli/IR-Sensor-Simulation-Engine)

<p align="right">(<a href="#readme-top">başa dön</a>)</p>



<!-- MARKDOWN LINKS & IMAGES -->
[CUDA-Shield]: https://img.shields.io/badge/cuda-000000?style=for-the-badge&logo=nvidia&logoColor=76B900
[CUDA-url]: https://developer.nvidia.com/cuda-toolkit
[OpenGL-Shield]: https://img.shields.io/badge/OpenGL-FFFFFF?style=for-the-badge&logo=opengl
[OpenGL-url]: https://www.opengl.org/
[Cpp-Shield]: https://img.shields.io/badge/C++-00599C?style=for-the-badge&logo=c%2B%2B&logoColor=white
[Cpp-url]: https://isocpp.org/
[ImGui-Shield]: https://img.shields.io/badge/Dear_ImGui-0A0A0A?style=for-the-badge&logo=cplusplus
[ImGui-url]: https://github.com/ocornut/imgui
