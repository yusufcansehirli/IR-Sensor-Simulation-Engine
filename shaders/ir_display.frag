// ir_display.frag — IR sensör görüntüsünü renklendirip ekrana yazar.
// Mimari belge §6.4 ile birebir örtüşür.
#version 450 core

uniform sampler2D u_ir_texture;   // uint16 sensör çıkışı (R kanalı float)
uniform int   u_colormap;         // 0=grayscale, 1=blackhot, 2=whitehot, 3=rainbow
uniform float u_gain;
uniform float u_level;

// Overlay mode
uniform bool  u_show_crosshair;
uniform bool  u_show_temp_scale;
uniform vec2  u_resolution;       // viewport boyutu

in  vec2 v_texcoord;
out vec4 frag_color;

// ─── Iron / Rainbow colormap (Iron: termal kameralarda yaygın) ───────────────
vec3 ironColormap(float v) {
    // Iron renk haritası: siyah → kırmızı → sarı → beyaz
    const vec3 c0 = vec3(0.0,  0.0,  0.0);
    const vec3 c1 = vec3(0.5,  0.0,  0.0);
    const vec3 c2 = vec3(1.0,  0.3,  0.0);
    const vec3 c3 = vec3(1.0,  0.85, 0.3);
    const vec3 c4 = vec3(1.0,  1.0,  1.0);

    if (v < 0.25) return mix(c0, c1, v / 0.25);
    if (v < 0.50) return mix(c1, c2, (v - 0.25) / 0.25);
    if (v < 0.75) return mix(c2, c3, (v - 0.50) / 0.25);
    return             mix(c3, c4, (v - 0.75) / 0.25);
}

vec3 rainbowColormap(float v) {
    float h = (1.0 - v) * 240.0 / 360.0;
    float r = abs(h * 6.0 - 3.0) - 1.0;
    float g = 2.0 - abs(h * 6.0 - 2.0);
    float b = 2.0 - abs(h * 6.0 - 4.0);
    return clamp(vec3(r, g, b), 0.0, 1.0);
}

vec3 applyColormap(float v, int mode) {
    v = clamp(v, 0.0, 1.0);
    switch (mode) {
        case 0:  return vec3(v);               // Grayscale
        case 1:  return vec3(1.0 - v);         // Blackhot
        case 2:  return vec3(v);               // Whitehot
        case 3:  return rainbowColormap(v);    // Rainbow
        case 4:  return ironColormap(v);       // Iron (termal)
        default: return vec3(v);
    }
}

// ─── Yatay sıcaklık skalası (ekran alt bandı) ────────────────────────────────
vec3 drawTempScale(vec2 uv) {
    if (uv.y > 0.96) {  // Alt %4
        float t = uv.x;
        return applyColormap(t, u_colormap);
    }
    return vec3(-1.0);  // Geçersiz → render
}

void main() {
    // ── Sıcaklık skalası overlay ─────────────────────────────────────────────
    if (u_show_temp_scale) {
        vec3 scale_color = drawTempScale(v_texcoord);
        if (scale_color.r >= 0.0) {
            frag_color = vec4(scale_color, 1.0);
            return;
        }
    }

    // ── IR görüntü ───────────────────────────────────────────────────────────
    float raw = texture(u_ir_texture, v_texcoord).r;  // zaten [0,1] normalize
    float dn  = (raw - u_level) * u_gain + 0.5;
    vec3  col = applyColormap(dn, u_colormap);

    // ── Crosshair overlay ────────────────────────────────────────────────────
    if (u_show_crosshair) {
        vec2 center = vec2(0.5, 0.5);
        float dx = abs(v_texcoord.x - center.x) * u_resolution.x;
        float dy = abs(v_texcoord.y - center.y) * u_resolution.y;
        if ((dx < 1.0 && dy < 15.0) || (dy < 1.0 && dx < 15.0)) {
            col = mix(col, vec3(0.0, 1.0, 0.0), 0.7);  // Yeşil crosshair
        }
    }

    frag_color = vec4(col, 1.0);
}
