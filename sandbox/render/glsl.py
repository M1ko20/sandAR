"""Sdílená knihovna GLSL pro hry: výška, barvy terénu, vrstevnice, stínování.

Hra napíše jen fragment shader s funkcí ``main``; ``RenderHost.program`` před něj
automaticky vloží tuto knihovnu. K dispozici jsou:

* ``v_uv``                 – souřadnice v bedně 0..1 (x: roh 1→2, y: roh 1→4)
* ``height_at(uv)``        – výška písku v metrech (hladká bikubická interpolace)
* ``height_grad(uv)``      – sklon (dh/dx, dh/dy)
* ``elevation_color(h)``   – barva hypsometrické mapy (s vodou pod hladinou)
* ``contour_line(h, interval, width_px)`` – 0..1 intenzita vrstevnice
* ``contours(h)``          – vrstevnice podle nastavení mapy (vedlejší + hlavní)
* ``hillshade(uv, k)``     – stínování reliéfu 0..1
* uniformy ``u_box`` (rozměr [m]), ``u_canvas`` (px), ``u_time`` (s) a nastavení mapy
"""

FULLSCREEN_VS = """
#version 330
in vec2 in_pos;
out vec2 v_uv;
void main() {
    v_uv = in_pos * 0.5 + 0.5;
    gl_Position = vec4(in_pos, 0.0, 1.0);
}
"""

LIB = """
#version 330
in vec2 v_uv;
out vec4 f_color;

uniform sampler2D u_height;
uniform vec2 u_box;
uniform vec2 u_canvas;
uniform float u_time;
uniform float u_elev_min;
uniform float u_elev_max;
uniform float u_water_level;
uniform float u_contour_interval;
uniform int u_major_every;

vec4 _cubic(float v) {
    vec4 n = vec4(1.0, 2.0, 3.0, 4.0) - v;
    vec4 s = n * n * n;
    float x = s.x;
    float y = s.y - 4.0 * s.x;
    float z = s.z - 4.0 * s.y + 6.0 * s.x;
    float w = 6.0 - x - y - z;
    return vec4(x, y, z, w) * (1.0 / 6.0);
}

float sample_bicubic(sampler2D tex, vec2 uv) {
    vec2 size = vec2(textureSize(tex, 0));
    vec2 inv = 1.0 / size;
    vec2 tc = uv * size - 0.5;
    vec2 f = fract(tc);
    tc -= f;
    vec4 xc = _cubic(f.x);
    vec4 yc = _cubic(f.y);
    vec4 c = tc.xxyy + vec2(-0.5, 1.5).xyxy;
    vec4 s = vec4(xc.xz + xc.yw, yc.xz + yc.yw);
    vec4 off = (c + vec4(xc.yw, yc.yw) / s) * inv.xxyy;
    float s0 = texture(tex, off.xz).r;
    float s1 = texture(tex, off.yz).r;
    float s2 = texture(tex, off.xw).r;
    float s3 = texture(tex, off.yw).r;
    float sx = s.x / (s.x + s.y);
    float sy = s.z / (s.z + s.w);
    return mix(mix(s3, s2, sx), mix(s1, s0, sx), sy);
}

float height_at(vec2 uv) { return sample_bicubic(u_height, uv); }

vec2 height_grad(vec2 uv) {
    vec2 e = 2.0 / vec2(textureSize(u_height, 0));
    float hx = height_at(uv + vec2(e.x, 0.0)) - height_at(uv - vec2(e.x, 0.0));
    float hy = height_at(uv + vec2(0.0, e.y)) - height_at(uv - vec2(0.0, e.y));
    return vec2(hx / (2.0 * e.x * u_box.x), hy / (2.0 * e.y * u_box.y));
}

const int N_LAND = 7;
const float LAND_T[N_LAND] = float[](0.0, 0.18, 0.38, 0.55, 0.72, 0.88, 1.0);
const vec3 LAND_C[N_LAND] = vec3[](
    vec3(0.16, 0.50, 0.24), vec3(0.42, 0.70, 0.32), vec3(0.80, 0.84, 0.45),
    vec3(0.95, 0.80, 0.42), vec3(0.80, 0.52, 0.28), vec3(0.58, 0.36, 0.24),
    vec3(0.97, 0.97, 0.97));

vec3 land_color(float t) {
    t = clamp(t, 0.0, 1.0);
    vec3 c = LAND_C[0];
    for (int i = 1; i < N_LAND; ++i) {
        float k = clamp((t - LAND_T[i - 1]) / (LAND_T[i] - LAND_T[i - 1]), 0.0, 1.0);
        c = mix(c, LAND_C[i], k);
    }
    return c;
}

vec3 water_color(float depth01) {
    return mix(vec3(0.35, 0.68, 0.98), vec3(0.03, 0.14, 0.48), clamp(depth01, 0.0, 1.0));
}

vec3 elevation_color(float h) {
    if (h < u_water_level) {
        return water_color((u_water_level - h) / max(u_water_level - u_elev_min, 1e-4));
    }
    return land_color((h - u_water_level) / max(u_elev_max - u_water_level, 1e-4));
}

// Vrstevnice leží na výškách (k + 0.5) * interval, aby uhlazený písek (h = 0)
// nebyl přesně na vrstevnici a šum nekreslil rozsypané čáry.
float contour_line(float h, float interval, float width_px) {
    float g = h / interval - 0.5;
    float d = abs(fract(g + 0.5) - 0.5);
    float w = max(fwidth(g), 1e-5);
    return 1.0 - smoothstep(0.5 * width_px * w, (0.5 * width_px + 1.0) * w, d);
}

// x = vedlejší vrstevnice, y = hlavní (zesílená) vrstevnice
vec2 contours(float h) {
    float minor = contour_line(h, u_contour_interval, 1.6);
    float major_interval = u_contour_interval * float(max(u_major_every, 1));
    float g = h / major_interval - 0.5 / float(max(u_major_every, 1));
    float d = abs(fract(g + 0.5) - 0.5);
    float w = max(fwidth(g), 1e-5);
    float major = (u_major_every > 1) ? 1.0 - smoothstep(1.6 * w, 2.6 * w, d) : 0.0;
    return vec2(minor, major);
}

float hillshade(vec2 uv, float exaggeration) {
    vec2 g = height_grad(uv) * exaggeration;
    vec3 n = normalize(vec3(-g.x, -g.y, 1.0));
    vec3 l = normalize(vec3(-1.0, -1.2, 1.6));
    return clamp(dot(n, l), 0.0, 1.0);
}

// Turbo-like barevná škála pro výškovou mapu
vec3 heat_color(float t) {
    t = clamp(t, 0.0, 1.0);
    const vec4 kr = vec4(0.13572138, 4.61539260, -42.66032258, 132.13108234);
    const vec4 kg = vec4(0.09140261, 2.19418839, 4.84296658, -14.18503333);
    const vec4 kb = vec4(0.10667330, 12.64194608, -60.58204836, 110.36276771);
    const vec2 kr2 = vec2(-152.94239396, 59.28637943);
    const vec2 kg2 = vec2(4.27729857, 2.82956604);
    const vec2 kb2 = vec2(-89.90310912, 27.34824973);
    vec4 v4 = vec4(1.0, t, t * t, t * t * t);
    vec2 v2 = v4.zw * v4.z;
    return clamp(vec3(dot(v4, kr) + dot(v2, kr2), dot(v4, kg) + dot(v2, kg2), dot(v4, kb) + dot(v2, kb2)), 0.0, 1.0);
}
"""
