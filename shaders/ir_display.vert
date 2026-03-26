// ir_display.vert — IR display vertex shader (fullscreen quad)
#version 450 core

layout(location = 0) in vec2 a_pos;
layout(location = 1) in vec2 a_uv;

out vec2 v_texcoord;

void main() {
    v_texcoord  = a_uv;
    gl_Position = vec4(a_pos, 0.0, 1.0);
}
