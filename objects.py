import glfw
from OpenGL.GL import *
import OpenGL.GL.shaders
import shaders
import utility
from engine_settings import *
import numpy as np
from objects import *
import pyrr
import math

class TextureObject:
	def __init__(self, resource, x, y, w=-1, h=-1, centered=True, alpha=True):
		self.shader = shaders.getImageShader()
		self.alpha = alpha

		self.texture = glGenTextures(1) # CORRECT
		glBindTexture(GL_TEXTURE_2D, self.texture)
		
		self.image, self.image_data = utility.loadImageData(resource)
		
		if self.alpha:
			glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, self.image.width, self.image.height, 0, GL_RGBA, GL_UNSIGNED_BYTE, self.image_data)
		else:
			glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, self.image.width, self.image.height, 0, GL_RGB, GL_UNSIGNED_BYTE, self.image_data)
		
		if w == -1 or h == -1:
			w = self.image.width
			h = self.image.height

		if centered:
			x -= self.image.width/2
			y -= self.image.height/2

		self.x1, self.x2 = x/DISPLAY_WIDTH*2-1, (x+w)/DISPLAY_WIDTH*2-1
		self.y1, self.y2 = y/DISPLAY_HEIGHT*2-1, (y+h)/DISPLAY_HEIGHT*2-1
		
		self.VAO = glGenBuffers(1)

		self.data = np.array([self.x1, self.y2, 0.0, 0.0,
						self.x1, self.y1, 0.0, 1.0,
						self.x2, self.y1, 1.0, 1.0,
						self.x2, self.y2, 1.0, 0.0], dtype=np.float32)
		

	def bindBuffers(self):
		glBindBuffer(GL_ARRAY_BUFFER, self.VAO)
		glBufferData(GL_ARRAY_BUFFER, self.data.nbytes, self.data, GL_DYNAMIC_DRAW)

		# Bind position data from model to shader
		position = glGetAttribLocation(self.shader, 'position')
		glVertexAttribPointer(position, 2, GL_FLOAT, GL_FALSE, 16, ctypes.c_void_p(0))
		glEnableVertexAttribArray(position)
	 
		# Bind texture coordinate data from model to shader
		texCoords = glGetAttribLocation(self.shader, "vertex_tex_coords")
		glVertexAttribPointer(texCoords, 2, GL_FLOAT, GL_FALSE, 16, ctypes.c_void_p(8))
		glEnableVertexAttribArray(texCoords)

	def render(self, alphaValue):
		glUseProgram(self.shader)
		self.bindBuffers()

		alphaLoc = glGetUniformLocation(self.shader, "alpha")
		glUniform1f(alphaLoc, alphaValue)

		if self.alpha:
			glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, self.image.width, self.image.height, 0, GL_RGBA, GL_UNSIGNED_BYTE, self.image_data)
		else:
			glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, self.image.width, self.image.height, 0, GL_RGB, GL_UNSIGNED_BYTE, self.image_data)

		glEnable(GL_BLEND);
		glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);  
		glDrawArrays(GL_QUADS, 0, 6)

class CartesianCoordsObject:
	def __init__(self, unitSize, widthUnits):
		self.shader = shaders.getCartShader()

		self.vertices = []

		self.widthUnits = widthUnits
		for i in range(self.widthUnits):
			self.vertices.append([])
			for j in range(self.widthUnits):
				self.vertices[i].append([unitSize*(i-self.widthUnits/2), 0, unitSize*(j-self.widthUnits/2)])

	def render(self, view, projection, model, transformation, alpha=1):
		glUseProgram(self.shader)

		view_loc = glGetUniformLocation(self.shader, "view")
		proj_loc = glGetUniformLocation(self.shader, "projection")
		model_loc = glGetUniformLocation(self.shader, "model")
		transform_loc = glGetUniformLocation(self.shader, "transform")

		alpha_loc = glGetUniformLocation(self.shader, "alpha")
		glUniform1f(alpha_loc, alpha)

		glUniformMatrix4fv(view_loc, 1, GL_FALSE, view)
		glUniformMatrix4fv(proj_loc, 1, GL_FALSE, projection)
		glUniformMatrix4fv(model_loc, 1, GL_FALSE, model)
		glUniformMatrix4fv(transform_loc, 1, GL_FALSE, transformation)

		#glColor3f(0,0,0)
		glBegin(GL_LINES)
		for i in range(self.widthUnits):
			for j in range(self.widthUnits):
				if i != self.widthUnits-1:
					v = self.vertices[i][j]
					glVertex3f(v[0], v[1], v[2])
					v = self.vertices[i+1][j]
					glVertex3f(v[0], v[1], v[2])
				if j != self.widthUnits-1:
					v = self.vertices[i][j]
					glVertex3f(v[0], v[1], v[2])
					v = self.vertices[i][j+1]
					glVertex3f(v[0], v[1], v[2])
		glEnd()

class PointCloudObject:
	def __init__(self):
		self.shader = shaders.getPointCloudShader()

		self.points = np.array([], dtype=np.float32)

		self.VAO = glGenVertexArrays(1)

	def addPoint(self, x, y, z, color):
		self.points = np.append(self.points, x)
		self.points = np.append(self.points, y)
		self.points = np.append(self.points, z)

		self.points = np.append(self.points, color[0])
		self.points = np.append(self.points, color[1])
		self.points = np.append(self.points, color[2])

	def dumpPoints(self, points):
		self.points = np.array(points, dtype=np.float32)

	def printPoints(self):
		print("Points:")
		for i in range(int(len(self.points)/6)):
			print(str(i) + ": " + str(self.points[i*6:i*6+6]))

	def bindBuffers(self):
		glBindBuffer(GL_ARRAY_BUFFER, self.VAO)
		glBufferData(GL_ARRAY_BUFFER, self.points.itemsize * len(self.points), self.points, GL_STATIC_DRAW)

		position = glGetAttribLocation(self.shader, 'position')
		glVertexAttribPointer(position, 3, GL_FLOAT, GL_FALSE, self.points.itemsize * 6, ctypes.c_void_p(0))
		glEnableVertexAttribArray(position)

		vertColor = glGetAttribLocation(self.shader, "vertColor")
		glVertexAttribPointer(vertColor, 3, GL_FLOAT, GL_FALSE, self.points.itemsize * 6, ctypes.c_void_p(12))
		glEnableVertexAttribArray(vertColor)

	def render(self, view, projection, model, transformation, lines=False, alpha=1):
		glUseProgram(self.shader)
		self.bindBuffers()

		# Bind matricies to shader
		view_loc = glGetUniformLocation(self.shader, "view")
		proj_loc = glGetUniformLocation(self.shader, "projection")
		model_loc = glGetUniformLocation(self.shader, "model")
		transform_loc = glGetUniformLocation(self.shader, "transform")

		glUniformMatrix4fv(view_loc, 1, GL_FALSE, view)
		glUniformMatrix4fv(proj_loc, 1, GL_FALSE, projection)
		glUniformMatrix4fv(model_loc, 1, GL_FALSE, model)
		glUniformMatrix4fv(transform_loc, 1, GL_FALSE, transformation)

		alpha_loc = glGetUniformLocation(self.shader, "alpha")
		glUniform1f(alpha_loc, alpha)

		if lines:
			glDrawArrays(GL_LINES, 0, len(self.points))
		else:
			glPointSize(2)
			glDrawArrays(GL_POINTS, 0, len(self.points))

class TargetModel:
	def __init__(self, verticies, verticiesPerLayer, totalLayers):
		self.shader = shaders.getTargetModelShader()
		
		# Generate a texture object and bind it to shader
		self.VAO = glGenBuffers(1)
		self.IAO = glGenBuffers(1)

		self.dumpVeritices(verticies, verticiesPerLayer, totalLayers)

	def dumpVeritices(self, verticies, verticiesPerLayer, totalLayers):
		# vertex = x, y, z, r, g, b
		data = verticies
		indices = []

		for li in range(totalLayers-1):
			for ri in range(verticiesPerLayer):
				p1 = li*verticiesPerLayer + ri
				p2 = li*verticiesPerLayer + (ri+1)%verticiesPerLayer
				p3 = (li+1)*verticiesPerLayer + ri
				p4 = (li+1)*verticiesPerLayer + (ri+1)%verticiesPerLayer

				indices.append(p1)
				indices.append(p2)
				indices.append(p3)

				indices.append(p4)
				indices.append(p2)
				indices.append(p3)

		for ri in range(verticiesPerLayer):
			p1 = (totalLayers-1)*verticiesPerLayer + ri
			p2 = (totalLayers-1)*verticiesPerLayer + (ri+1)%verticiesPerLayer
			p3 = int(len(verticies)/6) - 1

			indices.append(p1)
			indices.append(p2)
			indices.append(p3)

		self.verticies = np.array(data, dtype=np.float32)
		self.indices = np.array(indices, dtype=np.uint32)

	def bindBuffers(self): 
		glBindBuffer(GL_ARRAY_BUFFER, self.VAO)
		glBufferData(GL_ARRAY_BUFFER, self.verticies.itemsize * len(self.verticies), self.verticies, GL_STATIC_DRAW)
 
 
		# Bind index EAB (element array buffer)
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.IAO)
		glBufferData(GL_ELEMENT_ARRAY_BUFFER, self.indices.itemsize * len(self.indices), self.indices, GL_STATIC_DRAW)
 
 
		# Bind position data from model to shader
		position = glGetAttribLocation(self.shader, 'position')
		glVertexAttribPointer(position, 3, GL_FLOAT, GL_FALSE, self.verticies.itemsize * 6, ctypes.c_void_p(0))
		glEnableVertexAttribArray(position)
 
		# Bind texture coordinate data from model to shader
		color = glGetAttribLocation(self.shader, "color")
		glVertexAttribPointer(color, 3, GL_FLOAT, GL_FALSE, self.verticies.itemsize * 6, ctypes.c_void_p(12))
		glEnableVertexAttribArray(color)

	def render(self, view, projection, model, transformation):
		glUseProgram(self.shader)
		self.bindBuffers()

		# Bind matricies to shader
		view_loc = glGetUniformLocation(self.shader, "view")
		proj_loc = glGetUniformLocation(self.shader, "projection")
		model_loc = glGetUniformLocation(self.shader, "model")
		transform_loc = glGetUniformLocation(self.shader, "transform")

		glUniformMatrix4fv(view_loc, 1, GL_FALSE, view)
		glUniformMatrix4fv(proj_loc, 1, GL_FALSE, projection)
		glUniformMatrix4fv(model_loc, 1, GL_FALSE, model)
		glUniformMatrix4fv(transform_loc, 1, GL_FALSE, transformation)

		# Draw cube based on triangle polygons from loaded VBO's
		glDrawElements(GL_TRIANGLES, len(self.indices), GL_UNSIGNED_INT, None)