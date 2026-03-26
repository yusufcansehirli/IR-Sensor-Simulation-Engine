#version 450 core

in vec3 v_world_pos;
in vec3 v_world_normal;
flat in uint v_mat_id;

layout(location = 0) out vec4  gPosition;   // World-space position
layout(location = 1) out vec4  gNormal;     // World-space normal
layout(location = 2) out uvec4 gMaterial;   // Material ID (R kanalı uint)

void main() {
    gPosition = vec4(v_world_pos, 1.0);
    gNormal   = vec4(normalize(v_world_normal), 0.0);
    gMaterial = uvec4(v_mat_id, 0u, 0u, 0u);
}
