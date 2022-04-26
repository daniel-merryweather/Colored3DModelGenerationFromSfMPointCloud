from OpenGL.GL import *
import OpenGL.GL.shaders

imageVertexShader = """
#version 330

in vec4 position;
in vec2 vertex_tex_coords;

out vec2 fragment_tex_coords;

void main() {
	gl_Position = position;
	fragment_tex_coords = vertex_tex_coords;
}
"""

imageFragmentShader = """
#version 330
in vec2 fragment_tex_coords;

out vec4 output_color;

uniform sampler2D samplerTexture;
uniform float alpha;

void main() {
	output_color = texture2D(samplerTexture, fragment_tex_coords);
	//output_color.a = alpha;
	if(output_color.a < 0.01){
		discard;
	}else{
		output_color.a = alpha;
	}
}
"""

def getImageShader():
	shader = OpenGL.GL.shaders.compileProgram(
		OpenGL.GL.shaders.compileShader(imageVertexShader, GL_VERTEX_SHADER),
		OpenGL.GL.shaders.compileShader(imageFragmentShader, GL_FRAGMENT_SHADER))
	return shader

cartVertexShader = """
#version 330

in vec3 position;
//in vec3 vertColor;

//out vec4 fragColor;

uniform mat4 transform;
uniform mat4 view;
uniform mat4 model;
uniform mat4 projection;
 
void main() {
	gl_Position = projection * view * model * transform * vec4(position, 1.0f);
}
"""

cartFragmentShader = """
#version 330
 
//in vec4 fragColor;
out vec4 outColor;

uniform float alpha;

void main() {
	outColor = vec4(0.5, 0.5, 0.5, 1.0);
	outColor[3] = alpha;
}
"""

def getCartShader():
	shader = OpenGL.GL.shaders.compileProgram(
		OpenGL.GL.shaders.compileShader(cartVertexShader, GL_VERTEX_SHADER),
		OpenGL.GL.shaders.compileShader(cartFragmentShader, GL_FRAGMENT_SHADER))
	return shader


pointCloudVertexShader = """
#version 330

in vec3 position;
in vec3 vertColor;
 
out vec3 fragColor;

uniform mat4 projection;
uniform mat4 view;
uniform mat4 model;
uniform mat4 transform;
 
void main() {
	gl_Position = projection * view * model * transform * vec4(position, 1.0f);
	fragColor = vertColor;
	//fragColor = vec3(0.0,1.0,0.0);
}
"""

pointCloudFragmentShader = """
#version 330
 
in vec3 fragColor;
out vec4 outColor;

uniform float alpha;

void main() {
	outColor = vec4(fragColor, alpha);
}
"""

def getPointCloudShader():
	shader = OpenGL.GL.shaders.compileProgram(
		OpenGL.GL.shaders.compileShader(pointCloudVertexShader, GL_VERTEX_SHADER),
		OpenGL.GL.shaders.compileShader(pointCloudFragmentShader, GL_FRAGMENT_SHADER))
	return shader


targetModelVertexShader = """
#version 330

in vec3 position;
in vec3 color;

out vec3 fragColor;

uniform mat4 transform;
uniform mat4 view;
uniform mat4 model;
uniform mat4 projection;
 
void main() {
	gl_Position = projection * view * model * transform * vec4(position, 1.0f);
	fragColor = color;
}
"""

targetModelFragmentShader = """
#version 330

in vec3 fragColor;
out vec4 outColor;

void main() {
	outColor = vec4(fragColor, 1.0f);
}
"""

def getTargetModelShader():
	shader = OpenGL.GL.shaders.compileProgram(
		OpenGL.GL.shaders.compileShader(targetModelVertexShader, GL_VERTEX_SHADER),
		OpenGL.GL.shaders.compileShader(targetModelFragmentShader, GL_FRAGMENT_SHADER))
	return shader