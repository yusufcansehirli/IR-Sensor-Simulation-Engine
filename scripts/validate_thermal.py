#!/usr/bin/env python3
"""
validate_thermal.py
─────────────────────────────────────────────────────────────────────────────
IR Sensör Simülatörü — Fizik Doğrulama Scriptleri.
Mimari belge §10.3 ile örtüşür.

Grafikler:
  1. Planck kara cisim emisyonu (SWIR/MWIR/LWIR bantlarında)
  2. Bant-entegre radyans vs sıcaklık (karşılaştırma eğrileri)
  3. Sıcaklık hassasiyeti: dL/dT (NEdT tasarım referansı)
  4. Beer-Lambert atmosfer iletimi vs mesafe ve nem

Bağımlılıklar: numpy, matplotlib (scipy opsiyonel)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec

# ─── Fizik sabitleri ─────────────────────────────────────────────────────────
H_PLANCK = 6.62607015e-34   # J·s
C_LIGHT   = 2.99792458e+8   # m/s
K_BOLTZ   = 1.380649e-23    # J/K
STEFAN_B  = 5.670374419e-8  # W/m²/K⁴

# ─── Bant tanımları ──────────────────────────────────────────────────────────
BANDS = {
    'SWIR': {'min': 1.0,  'max': 1.7,  'center': 1.55, 'color': 'blue',   'eps': 0.85},
    'MWIR': {'min': 3.5,  'max': 4.9,  'center': 4.0,  'color': 'orange', 'eps': 0.92},
    'LWIR': {'min': 8.5,  'max': 11.5, 'center': 10.0, 'color': 'red',    'eps': 0.95},
}

def planck(T, lam_um):
    """Planck kara cisim spektral radyansı [W/m²/sr/μm]"""
    lam = lam_um * 1e-6
    c1  = 2.0 * H_PLANCK * C_LIGHT**2
    c2  = (H_PLANCK * C_LIGHT) / (K_BOLTZ * T)
    return (c1 / lam**5) / (np.exp(c2 / lam) - 1.0) * 1e-6  # /m → /μm

def band_radiance(T, eps, lam_min, lam_max, n=100):
    """Simpson kuralıyla bant-entegre radyans [W/m²/sr]"""
    lams = np.linspace(lam_min, lam_max, n)
    L    = eps * planck(T, lams)
    return np.trapz(L, lams)

def beer_lambert_transmission(range_km, alpha_per_km):
    """Beer-Lambert: τ = exp(-α × r)"""
    return np.exp(-alpha_per_km * range_km)

# ─── 1. Planck Spektrumu ─────────────────────────────────────────────────────
def plot_planck_spectra():
    fig, ax = plt.subplots(figsize=(10, 5))

    lam = np.linspace(0.5, 14.0, 1000)
    temps   = [250, 300, 350, 400]
    colors  = ['#2196F3', '#4CAF50', '#FF9800', '#F44336']

    for T, col in zip(temps, colors):
        L = planck(T, lam)
        ax.semilogy(lam, L, color=col, label=f'T = {T} K ({T-273:.0f}°C)',
                    linewidth=2)

    # Bant vurguları
    for band_name, bd in BANDS.items():
        ax.axvspan(bd['min'], bd['max'], alpha=0.12, color=bd['color'],
                   label=f'{band_name} ({bd["min"]}–{bd["max"]} μm)')
        ax.axvline(bd['center'], color=bd['color'], linestyle='--',
                   alpha=0.5, linewidth=0.8)

    ax.set_xlabel('Dalga Boyu [μm]', fontsize=12)
    ax.set_ylabel('Spektral Radyans [W/m²/sr/μm]', fontsize=12)
    ax.set_title('Planck Kara Cisim Emisyonu — Farlı Sıcaklıklar', fontsize=14)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, which='both', alpha=0.3)
    ax.set_xlim(0.5, 14.0)
    ax.set_ylim(1e-5, 1e7)
    return fig

# ─── 2. Bant Radyansı vs Sıcaklık ───────────────────────────────────────────
def plot_radiance_vs_temperature():
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    T_range = np.linspace(260, 420, 200)

    for ax, (band_name, bd) in zip(axes, BANDS.items()):
        L_vals = [band_radiance(T, bd['eps'], bd['min'], bd['max']) for T in T_range]
        ax.plot(T_range, L_vals, color=bd['color'], linewidth=2.5)
        ax.axvline(300, color='gray', linestyle=':', linewidth=1.0, label='300 K ref')

        # Türev (NEdT hassasiyet)
        dL = np.gradient(L_vals, T_range)
        ax2 = ax.twinx()
        ax2.plot(T_range, dL, color='purple', linestyle='--',
                 linewidth=1.2, alpha=0.7, label='dL/dT')
        ax2.set_ylabel('dL/dT [W/m²/sr/K]', color='purple', fontsize=9)

        ax.set_xlabel('Yüzey Sıcaklığı [K]', fontsize=11)
        ax.set_ylabel(f'L_{band_name} [W/m²/sr]', fontsize=11)
        ax.set_title(f'{band_name} ({bd["min"]}–{bd["max"]} μm)\nε = {bd["eps"]}',
                     fontsize=12)
        ax.grid(True, alpha=0.3)

        # 300K değerini işaretle
        L_300 = band_radiance(300, bd['eps'], bd['min'], bd['max'])
        ax.axhline(L_300, color='red', linestyle=':', linewidth=0.8)
        ax.text(265, L_300 * 1.1, f'{L_300:.3e}', fontsize=8, color='red')
        ax.legend(loc='upper left', fontsize=8)

    fig.suptitle('Bant-Entegre Radyans vs Yüzey Sıcaklığı', fontsize=14,
                 fontweight='bold')
    plt.tight_layout()
    return fig

# ─── 3. NEdT Hassasiyet Analizi ─────────────────────────────────────────────
def plot_nedt_sensitivity():
    fig, ax = plt.subplots(figsize=(9, 5))

    T_range = np.linspace(270, 380, 200)
    nedt_targets = {'MWIR ideal': (1, 20), 'LWIR tipik': (2, 50),
                    'MWIR iyi': (1, 8)}

    for label, (band_idx, nedt_mk) in nedt_targets.items():
        band_name = list(BANDS.keys())[band_idx]
        bd = BANDS[band_name]
        dL_vals = []
        for T in T_range:
            L1 = band_radiance(T - 0.1, bd['eps'], bd['min'], bd['max'])
            L2 = band_radiance(T + 0.1, bd['eps'], bd['min'], bd['max'])
            dLdT = (L2 - L1) / 0.2
            # NEdT = σ_noise / (dL/dT)
            # Tersinden: σ_radiance = NEdT * dL/dT
            sigma_rad = (nedt_mk * 1e-3) * dLdT
            dL_vals.append(sigma_rad)
        ax.plot(T_range, dL_vals, linewidth=2,
                label=f'{label} (NEdT={nedt_mk}mK)')

    ax.set_xlabel('Hedef Sıcaklık [K]', fontsize=12)
    ax.set_ylabel('σ_radiance = NEdT × dL/dT [W/m²/sr]', fontsize=11)
    ax.set_title('NEdT Eşdeğer Radyans Gürültüsü', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    return fig

# ─── 4. Atmosfer İletimi ─────────────────────────────────────────────────────
def plot_atmospheric_transmission():
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Mesafe bağımlılığı
    ranges_km = np.linspace(0.1, 10, 200)
    alpha = {'SWIR': 0.10, 'MWIR': 0.15, 'LWIR': 0.05}
    for band_name, a in alpha.items():
        bd = BANDS[band_name]
        tau = beer_lambert_transmission(ranges_km, a)
        axes[0].plot(ranges_km, tau * 100, color=bd['color'],
                     linewidth=2.5, label=f'{band_name} (α={a}/km)')

    axes[0].set_xlabel('Menzil [km]', fontsize=12)
    axes[0].set_ylabel('İletim [%]', fontsize=12)
    axes[0].set_title('Atmosferik İletim vs Menzil\n(Analitik Beer-Lambert)', fontsize=12)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(0, 105)

    # Nem bağımlılığı @ 2 km
    humid_range = np.linspace(10, 95, 100)
    r_fixed = 2.0
    for band_name, a_base in alpha.items():
        bd = BANDS[band_name]
        tau_vals = []
        for H in humid_range:
            a_eff = a_base * (1.0 + 0.5 * (H - 50) / 50)
            tau_vals.append(beer_lambert_transmission(r_fixed, a_eff) * 100)
        axes[1].plot(humid_range, tau_vals, color=bd['color'],
                     linewidth=2.5, label=band_name)

    axes[1].set_xlabel('Bağıl Nem [%]', fontsize=12)
    axes[1].set_ylabel('İletim [%]', fontsize=12)
    axes[1].set_title(f'İletim vs Nem (r = {r_fixed} km)', fontsize=12)
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(0, 105)

    fig.suptitle('Atmosferik İletim Analizi', fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig

# ─── 5. Referans değerler çıktısı ───────────────────────────────────────────
def print_reference_values():
    print("\n" + "="*60)
    print("REFERANS DEĞERLER — T = 300 K")
    print("="*60)
    for band_name, bd in BANDS.items():
        L = band_radiance(300, bd['eps'], bd['min'], bd['max'])
        L1= band_radiance(299.9, bd['eps'], bd['min'], bd['max'])
        L2= band_radiance(300.1, bd['eps'], bd['min'], bd['max'])
        dL= (L2 - L1) / 0.2
        print(f"  {band_name} ({bd['min']}–{bd['max']} μm), ε={bd['eps']}:")
        print(f"    L(300K)   = {L:.4e} W/m²/sr")
        print(f"    dL/dT     = {dL:.4e} W/m²/sr/K")
        print(f"    NEdT=25mK → σ_L = {dL*0.025:.4e} W/m²/sr")

    # Planck @ 300K, 10μm — doğrulama değeri
    L_pt = planck(300, 10.0)
    print(f"\n  Planck(300K, 10μm) = {L_pt:.4f} W/m²/sr/μm")
    print(f"  [Beklenen: ~9.5 W/m²/sr/μm — ±5% doğrulama hedefi]")

# ─── Ana işlev ───────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print_reference_values()

    fig1 = plot_planck_spectra()
    fig2 = plot_radiance_vs_temperature()
    fig3 = plot_nedt_sensitivity()
    fig4 = plot_atmospheric_transmission()

    fig1.savefig('validation_planck_spectra.png', dpi=150, bbox_inches='tight')
    fig2.savefig('validation_radiance_vs_temp.png', dpi=150, bbox_inches='tight')
    fig3.savefig('validation_nedt_sensitivity.png', dpi=150, bbox_inches='tight')
    fig4.savefig('validation_atm_transmission.png', dpi=150, bbox_inches='tight')

    print("\n[validate_thermal.py] Grafikler kaydedildi:")
    print("  validation_planck_spectra.png")
    print("  validation_radiance_vs_temp.png")
    print("  validation_nedt_sensitivity.png")
    print("  validation_atm_transmission.png")
    plt.show()
