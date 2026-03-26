#version 450 core

layout(location = 0) in vec3 a_position;
layout(location = 1) in vec3 a_normal;
layout(location = 2) in uint a_mat_id;

uniform mat4 u_view;
uniform mat4 u_proj;

out vec3 v_world_pos;
out vec3 v_world_normal;
flat out uint v_mat_id;

void main() {
    v_world_pos       = a_position; // Model matrix is identity
    v_world_normal    = a_normal;
    v_mat_id          = a_mat_id;
    gl_Position       = u_proj * u_view * vec4(v_world_pos, 1.0);
}
