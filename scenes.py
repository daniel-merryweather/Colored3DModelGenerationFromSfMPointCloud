import glfw
from OpenGL.GL import *
import OpenGL.GL.shaders
import shaders
import math
from objects import *
import utility
import random
import pyrr

class IntroScene:
	def __init__(self):
		self.shouldEnd = False

		self.tex = TextureObject("EngineResources/ua-logo-white.png", DISPLAY_WIDTH/2, DISPLAY_HEIGHT/2, alpha=True)

		# TEXTURE SAMPLING SETTINGS
		# Set the texture wrapping parameters (texture coords will loop back if beyond image domain)
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
	 
		# Set texture filtering parameters (linear filtering, so zero antialiasing on texture)
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
	 
		glClearColor(0,0,0, 1.0)
		glEnable(GL_DEPTH_TEST)

		glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
		glEnable(GL_BLEND);

		self.startSequence()

	def startSequence(self):
		self.sequence = 0
		self.start_time = glfw.get_time() + 1

	def sceneLoop(self, window):
		self.run_time = glfw.get_time()-self.start_time
		self.sequence = math.floor(self.run_time)

		glClearDepth(1.0)
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

		if self.sequence == 0:
			scalar = math.sin(self.run_time%1*math.pi/2)
			glClearColor(UOFA_GREEN[0]*scalar, UOFA_GREEN[1]*scalar, UOFA_GREEN[2]*scalar, 1.0)
		elif self.sequence == 1 or self.sequence == 2:
			glClearColor(UOFA_GREEN[0], UOFA_GREEN[1], UOFA_GREEN[2], 1.0)
			glDrawArrays(GL_QUADS, 0, 6)
			self.tex.render(math.sin((self.run_time-1)*math.pi/2))
		elif self.sequence == 3:
			scalar = math.sin(self.run_time%1*math.pi/2 + math.pi/2)
			glClearColor(UOFA_GREEN[0]+(1-UOFA_GREEN[0])*(1-scalar), UOFA_GREEN[1]+(1-UOFA_GREEN[1])*(1-scalar), UOFA_GREEN[2]+(1-UOFA_GREEN[2])*(1-scalar), 1.0)
		elif self.sequence > 4:
			self.shouldEnd = True

	def linkCallbackFunctions(self, window):
		glfw.set_key_callback(window, self.inputEvent)

	def inputEvent(self, window, key, scancode, action, mods):
		if action == glfw.PRESS:
			if key == glfw.KEY_ESCAPE: # Creature comfort, escape safely terminates the program
				glfw.set_window_should_close(window, GL_TRUE)
			elif key == glfw.KEY_SPACE:
				self.shouldEnd = True

class CubeScene:
	def __init__(self):
		self.shouldEnd = False
		self.start_time = glfw.get_time()

		self.view = pyrr.matrix44.create_from_translation(pyrr.Vector3([0,0,-5])) # Projector global offset
		self.projection = pyrr.matrix44.create_perspective_projection(80, DISPLAY_WIDTH/DISPLAY_HEIGHT, 0.1, 50.0) # Sets projection matrix with 20 degree FOV
		self.model = pyrr.matrix44.create_from_translation(pyrr.Vector3([0,0,0])) # Global origin
		self.transformation = utility.createTransformationMatrix(0,0,0, 0.5+0.25*math.sin(glfw.get_time()/6),glfw.get_time()/5)

		self.cart = CartesianCoordsObject(2, 25)

		self.postssmode = 22
		self.pc = PointCloudObject()
		self.pc2 = PointCloudObject()
		self.pc3 = PointCloudObject()
		self.targetModel = None

		self.xScroll = 0
		self.yScroll = 0

		self.model_x_rot = 0
		self.model_y_rot = 0
		self.model_z_rot = 0
		self.model_x_bounds = [-10,10]
		self.model_y_bounds = [-10,10]
		self.model_z_bounds = [-10,10]

		self.mode = 0
		
		self.cartesianGuide = PointCloudObject()
		guide = np.array([
			0,0,0,1,0,0,
			1,0,0,1,0,0,
			0,0,0,0,1,0,
			0,1,0,0,1,0,
			0,0,0,0,0,1,
			0,0,1,0,0,1])

		self.cartesianGuide.dumpPoints(guide)

		self.line = PointCloudObject()

		self.points3d = utility.read_points3D_binary("EngineResources/colmapData/points3D.bin")
		print("Number of points:")
		print(len(self.points3d))

		newPoints = np.zeros(len(self.points3d.keys())*6)

		points = np.array(list(self.points3d.values()))

		newPoints[0::6] = points[:,0,0]
		newPoints[1::6] = points[:,0,1]
		newPoints[2::6] = points[:,0,2]

		newPoints[3::6] = points[:,1,0]/255
		newPoints[4::6] = points[:,1,1]/255
		newPoints[5::6] = points[:,1,2]/255

		self.originPoints = newPoints.copy()

		print("Formatted.")

		self.cameras = utility.read_images_binary("EngineResources/colmapData/images.bin")
		self.cameraIndex = 1

		image_points, name = utility.backprojectPoints(self.points3d, self.cameras[self.cameraIndex])
		print("Name = " + name)

		x_rot, y_rot = utility.loadCameraAngle("EngineResources/colmapData/transforms.json")

		rotationMatrix = utility.createTransformationMatrix(xRotation=y_rot+math.pi, yRotation=x_rot)

		self.transformations = []
		newPoints = utility.transformPoints(newPoints, rotationMatrix)
		self.transformations.append(rotationMatrix)
		print("Rotation applied.")

		x_avg = np.average(newPoints[0::6])
		y_avg = np.average(newPoints[1::6])
		z_avg = np.average(newPoints[2::6])

		translationMatrix = np.array([
			[1,0,0,-x_avg],
			[0,1,0,-y_avg],
			[0,0,1,-z_avg],
			[0,0,0,1]
			])

		newPoints = utility.transformPoints(newPoints, translationMatrix)
		self.transformations.append(translationMatrix)
		print("Transformation applied.")

		#newPoints = utility.reverseTransformations(newPoints, self.transformations)

		self.pc.dumpPoints(newPoints)

		# TEXTURE SAMPLING SETTINGS
		# Set the texture wrapping parameters (texture coords will loop back if beyond image domain)
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
	 
		# Set texture filtering parameters (linear filtering, so zero antialiasing on texture)
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
	 
		glClearColor(1,1,1, 1.0)
		glEnable(GL_DEPTH_TEST)

	def sceneLoop(self, window):

		self.view = pyrr.matrix44.create_from_translation(pyrr.Vector3([0,0,-5])) # Projector global offset
		self.projection = pyrr.matrix44.create_perspective_projection(80, DISPLAY_WIDTH/DISPLAY_HEIGHT, 0.1, 50.0) # Sets projection matrix with 20 degree FOV
		self.model = pyrr.matrix44.create_from_translation(pyrr.Vector3([0,0,0])) # Global origin


		if self.mode == 1:
			self.projection = pyrr.matrix44.create_orthogonal_projection(-19.2,19.2, -10.8, 10.8, 1, 1000)
			self.object = utility.createTransformationMatrix(0,0,0, 0,0,self.model_y_rot/100)
			self.transformation = utility.createTransformationMatrix(0,0,0, math.pi/2,0,0)
		elif self.mode == 2:
			self.mode += 1
			rotation = utility.createTransformationMatrix(0,0,0, 0, -self.model_y_rot/100, 0)
			self.pc.dumpPoints(utility.transformPoints(self.pc.points, rotation))
			self.transformations.append(rotation)
			print("Y Rotation Applied.")


		elif self.mode == 3:
			self.projection = pyrr.matrix44.create_orthogonal_projection(-19.2,19.2, -10.8, 10.8, 1, 1000)
			self.object = utility.createTransformationMatrix(0,0,0, 0,0,self.model_x_rot/100)
			self.transformation = utility.createTransformationMatrix(0,0,0, 0,math.pi/2,0)
		elif self.mode == 4:
			self.mode += 1
			rotation = utility.createTransformationMatrix(0,0,0, self.model_x_rot/100, 0, 0)
			self.pc.dumpPoints(utility.transformPoints(self.pc.points, rotation))
			self.transformations.append(rotation)
			print("X Rotation Applied.")


		elif self.mode == 5:
			self.projection = pyrr.matrix44.create_orthogonal_projection(-19.2,19.2, -10.8, 10.8, 1, 1000)
			self.object = utility.createTransformationMatrix(0,0,0, 0,0,self.model_z_rot/100)
			self.transformation = utility.createTransformationMatrix(0,0,0, 0,0,0)
		elif self.mode == 6:
			self.mode += 1
			rotation = utility.createTransformationMatrix(0,0,0, 0, 0, -self.model_z_rot/100)
			self.pc.dumpPoints(utility.transformPoints(self.pc.points, rotation))
			self.transformations.append(rotation)
			print("Z Rotation Applied.")


		elif self.mode == 7:
			self.projection = pyrr.matrix44.create_orthogonal_projection(-19.2,19.2, -10.8, 10.8, 1, 1000)
			line = np.array([
				self.model_z_bounds[0]/19.2, 1,0,	1,0,0,
				self.model_z_bounds[0]/19.2,-1,0,	1,0,0])
			self.line.dumpPoints(line)
			self.transformation = utility.createTransformationMatrix(0,0,0, 0,-math.pi/2,0)
		elif self.mode == 8:
			self.projection = pyrr.matrix44.create_orthogonal_projection(-19.2,19.2, -10.8, 10.8, 1, 1000)
			line = np.array([
				self.model_z_bounds[1]/19.2, 1,0,	1,0,0,
				self.model_z_bounds[1]/19.2,-1,0,	1,0,0])
			self.line.dumpPoints(line)
			self.transformation = utility.createTransformationMatrix(0,0,0, 0,-math.pi/2,0)
		elif self.mode == 9:
			self.mode += 1
			points = self.pc.points
			for i in range(int(len(points)/6)):
				if points[i*6+2] < self.model_z_bounds[0] or points[i*6+2] > self.model_z_bounds[1]:
					points[i*6+0] = 0
					points[i*6+1] = 0
					points[i*6+2] = 0
			self.pc.dumpPoints(points)
			print("Z Bounds Clipped.")


		elif self.mode == 10:
			self.projection = pyrr.matrix44.create_orthogonal_projection(-19.2,19.2, -10.8, 10.8, 1, 1000)
			line = np.array([
				self.model_x_bounds[0]/19.2, 1,0,	1,0,0,
				self.model_x_bounds[0]/19.2,-1,0,	1,0,0])
			self.line.dumpPoints(line)
			self.transformation = utility.createTransformationMatrix(0,0,0, 0,0,0)
		elif self.mode == 11:
			self.projection = pyrr.matrix44.create_orthogonal_projection(-19.2,19.2, -10.8, 10.8, 1, 1000)
			line = np.array([
				self.model_x_bounds[1]/19.2, 1,0,	1,0,0,
				self.model_x_bounds[1]/19.2,-1,0,	1,0,0])
			self.line.dumpPoints(line)
			self.transformation = utility.createTransformationMatrix(0,0,0, 0,0,0)
		elif self.mode == 12:
			self.mode += 1
			points = self.pc.points
			for i in range(int(len(points)/6)):
				if points[i*6+0] < self.model_x_bounds[0] or points[i*6+0] > self.model_x_bounds[1]:
					points[i*6+0] = 0
					points[i*6+1] = 0
					points[i*6+2] = 0
			self.pc.dumpPoints(points)
			print("X Bounds Clipped.")


		elif self.mode == 13:
			self.projection = pyrr.matrix44.create_orthogonal_projection(-19.2,19.2, -10.8, 10.8, 1, 1000)
			line = np.array([
				1,self.model_y_bounds[0]/10.8,0,	1,0,0,
				-1,self.model_y_bounds[0]/10.8,0,	1,0,0])
			self.line.dumpPoints(line)
			self.transformation = utility.createTransformationMatrix(0,0,0, 0,-math.pi/2,0)
		elif self.mode == 14:
			self.projection = pyrr.matrix44.create_orthogonal_projection(-19.2,19.2, -10.8, 10.8, 1, 1000)
			line = np.array([
				1,self.model_y_bounds[1]/10.8,0,	1,0,0,
				-1,self.model_y_bounds[1]/10.8,0,	1,0,0])
			self.line.dumpPoints(line)
			self.transformation = utility.createTransformationMatrix(0,0,0, 0,-math.pi/2,0)
		elif self.mode == 15:
			self.mode += 1
			points = self.pc.points
			for i in range(int(len(points)/6)):
				if points[i*6+1] < self.model_y_bounds[0] or points[i*6+1] > self.model_y_bounds[1]:
					points[i*6+0] = 0
					points[i*6+1] = 0
					points[i*6+2] = 0
			print("Y Bounds Clipped.")
			self.line.dumpPoints(np.array([]))
			print("Pre-Clipping Points: " + str(int(len(points)/6)))
			points = utility.removeZeroPoints(points)
			print("Post-Clipping Points: " + str(int(len(points)/6)))
			self.pc.dumpPoints(points)

		elif self.mode == 16:
			print("Calculating Hash...", end='')
			self.object = utility.createTransformationMatrix(0,0,0, 0,0,0)
			self.transformation = utility.createTransformationMatrix(0,0,0, self.yScroll/100, self.xScroll/100, 0)
			self.pointHash = utility.binaryHashFrom3DPoints(self.pc.points, unitSize=0.1)
			self.mode += 1
			print("DONE.")

		elif self.mode == 18:
			self.pc.dumpPoints(utility.filterDensity(self.pc.points, self.pointHash, 8))
			self.mode += 1
			self.object = utility.createTransformationMatrix(0,0,0, 0,0,0)
			self.transformation = utility.createTransformationMatrix(0,0,0, self.yScroll/100, self.xScroll/100, 0)
			print("Coarse Filter Applied.")
			print("Total Points After Filter: " + str(int(len(self.pc.points)/6)))

		elif self.mode == 20:
			print("Calculating Hash...", end='')
			self.object = utility.createTransformationMatrix(0,0,0, 0,0,0)
			self.transformation = utility.createTransformationMatrix(0,0,0, self.yScroll/100, self.xScroll/100, 0)
			self.pointHash = utility.binaryHashFrom3DPoints(self.pc.points, unitSize=0.1)
			self.mode += 1
			print("DONE.")
			self.postssmode = 22 + len(self.pointHash[1])

		elif self.mode >= 22 and self.mode < self.postssmode:
			layer = utility.extractRangeFrom3DBinaryHash(self.pointHash, 0, len(self.pointHash[0]), self.mode-22, self.mode-21, 0, len(self.pointHash[2]))
			n = 3000
			if self.mode < 25:
				n = 10000
			if len(self.pc.points[layer*6]) > 300:
				superSamples = np.zeros(n*6)
				for i in range(n):
					xlayerValues = self.pc.points[layer*6+0]
					ylayerValues = self.pc.points[layer*6+1]
					zlayerValues = self.pc.points[layer*6+2]

					randomIndex1 = random.choice(range(len(xlayerValues)))
					randomIndex2 = random.choice(range(len(xlayerValues)))

					randomPoint1 = np.array([xlayerValues[randomIndex1], ylayerValues[randomIndex1], zlayerValues[randomIndex1]])
					randomPoint2 = np.array([xlayerValues[randomIndex2], ylayerValues[randomIndex2], zlayerValues[randomIndex2]])

					superSamplePoint = randomPoint1 + np.multiply((randomPoint2 - randomPoint1), np.random.rand(3))

					superSamples[i*6+0] = superSamplePoint[0]
					superSamples[i*6+1] = superSamplePoint[1]
					superSamples[i*6+2] = superSamplePoint[2]
					superSamples[i*6+3] = 0
					superSamples[i*6+4] = 1
					superSamples[i*6+5] = 0

				points = np.concatenate([self.pc.points, superSamples])
				self.pc.dumpPoints(points)
			self.mode += 1

		elif self.mode == self.postssmode+2:
			print("Calculating Hash...", end='')
			self.object = utility.createTransformationMatrix() # Global origin
			self.transformation = utility.createTransformationMatrix(0,0,0, self.yScroll/100, self.xScroll/100)
			self.pointHash = utility.binaryHashFrom3DPoints(self.pc.points, unitSize=0.05)
			self.mode += 1
			print("DONE.")

		elif self.mode == self.postssmode+4:
			self.object = utility.createTransformationMatrix() # Global origin
			self.transformation = utility.createTransformationMatrix(0,0,0, self.yScroll/100, self.xScroll/100)
			self.pc.dumpPoints(utility.filterDensity(self.pc.points, self.pointHash, 5))
			self.mode += 1
			print("Fine Filter Applied.")
			print("Total Points After Filter: " + str(int(len(self.pc.points)/6)))

		elif self.mode == self.postssmode+6:
			self.object = utility.createTransformationMatrix() # Global origin
			self.transformation = utility.createTransformationMatrix(0,0,0, self.yScroll/100, self.xScroll/100)

			self.placementLevel = 0
			self.placementMax = 43
			self.rotationIndex = 0
			self.placements = []
			self.mode += 1

			xlayerValues = self.pc.points[0::6]
			ylayerValues = self.pc.points[1::6]
			zlayerValues = self.pc.points[2::6]
			self.ylower = np.min(ylayerValues)
			self.cloudPoints = np.array([xlayerValues, ylayerValues, zlayerValues])
			unitSize = 0.04
			self.layer = self.cloudPoints.T[np.where(
						(self.cloudPoints[1] >= self.ylower+(self.placementLevel-.5)*unitSize) &
						(self.cloudPoints[1] <= self.ylower+(self.placementLevel+1.5)*unitSize)
						)].T

		elif self.mode == self.postssmode+8:

			unitSize = 0.04
			self.rotations = 60

			if self.placementLevel < self.placementMax:
				if self.rotationIndex >= self.rotations:
					self.rotationIndex = 0
					self.placementLevel += 1
					self.layer = self.cloudPoints.T[np.where(
						(self.cloudPoints[1] >= self.ylower+(self.placementLevel-.5)*unitSize) &
						(self.cloudPoints[1] <= self.ylower+(self.placementLevel+1.5)*unitSize)
						)].T
				else:
					placementOffset = np.array([2,0,0,1]).dot(pyrr.Matrix44.from_y_rotation(self.rotationIndex/self.rotations*math.pi*2))
					placementPoint = np.array([np.average(self.layer[0]), self.ylower+(self.placementLevel+0.5)*unitSize, np.average(self.layer[2]), 0]) + placementOffset

					count = 0
					while utility.heuristicDensity(self.layer, placementPoint[0:3], 0.05) < 10:
						count += 1
						placementOffset *= 0.95
						placementPoint = np.array([np.average(self.layer[0]), self.ylower+(self.placementLevel+0.5)*unitSize, np.average(self.layer[2]), 0]) + placementOffset
						if count > 50:
							break
					placementOffset *= 0.95
					placementPoint = np.array([np.average(self.layer[0]), self.ylower+(self.placementLevel+0.5)*unitSize, np.average(self.layer[2]), 0]) + placementOffset

					self.placements = np.append(self.placements, [placementPoint[0], placementPoint[1], placementPoint[2], placementPoint[0], placementPoint[1], placementPoint[2]])
					self.pc3.dumpPoints(self.placements.flatten())
					self.rotationIndex += 1
			else:
				
				xvalues = self.pc3.points[0::6]
				yvalues = self.pc3.points[1::6]
				zvalues = self.pc3.points[2::6]

				lastpoint = [np.average(xvalues[len(xvalues)-self.rotations:len(xvalues)]),
							np.average(yvalues[len(xvalues)-self.rotations:len(xvalues)]),
							np.average(zvalues[len(xvalues)-self.rotations:len(xvalues)]),
							np.average(xvalues[len(xvalues)-self.rotations:len(xvalues)]),
							np.average(yvalues[len(xvalues)-self.rotations:len(xvalues)]),
							np.average(zvalues[len(xvalues)-self.rotations:len(xvalues)])]
				self.placements = np.append(self.placements, lastpoint)
				self.pc3.dumpPoints(self.placements.flatten())

				self.targetModel = TargetModel(self.pc3.points, self.rotations, self.placementMax)
				self.mode += 1

		elif self.mode == self.postssmode+10:
			self.object = utility.createTransformationMatrix(0,0,0, 0,0,0) # Global origin
			self.transformation = utility.createTransformationMatrix(0,0,0, 0,0,0)
			self.pc3.dumpPoints(utility.reverseTransformations(self.pc3.points, self.transformations))
			self.oldpoints = self.pc3.points.copy()
			x_avg = np.average(self.oldpoints[0::6])
			y_avg = np.average(self.oldpoints[1::6])
			z_avg = np.average(self.oldpoints[2::6])
			translationMatrix = np.array([
				[1,0,0,-x_avg],
				[0,1,0,-y_avg],
				[0,0,1,-z_avg],
				[0,0,0,1]
				])
			self.oldpoints = utility.transformPoints(self.oldpoints, translationMatrix)
			self.mode += 1
		elif self.mode == self.postssmode+12:
			image_points, name = utility.backprojectPoints2(self.pc3.points, self.cameras[self.cameraIndex])
			self.cameraTransform = utility.transformationFromCamera(self.cameras[self.cameraIndex])
			image = np.array(utility.loadImage("EngineResources/colmapData/drill_full/"+name))
			imps = image_points.reshape(-1,6)
			for pi in range(len(imps)):
				imp = imps[pi]
				x = imp[0]
				y = -imp[1]
				x += 1
				x = x/2
				y += 1
				y = y/2
				x = int(x*1920)
				y = int(y*1080)
				color = image[y][x]
				self.oldpoints[pi*6+3] = color[0]/255
				self.oldpoints[pi*6+4] = color[1]/255
				self.oldpoints[pi*6+5] = color[2]/255
			self.targetModel = TargetModel(self.oldpoints, self.rotations, self.placementMax)
			self.mode += 1
		elif self.mode == self.postssmode+13:
			self.object = utility.createTransformationMatrix(0,0,0, 0,0,0) # Global origin
			self.transformation = self.cameraTransform
		else:
			self.object = utility.createTransformationMatrix(0,0,0, 0,0,0) # Global origin
			self.transformation = utility.createTransformationMatrix(0,0,0, self.yScroll/100, self.xScroll/100, 0)

		glClearDepth(1.0)
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

		glClearColor(1, 1, 1, 1.0)

		alpha = 1
		if glfw.get_time()-self.start_time < 5:
			alpha = math.sin((glfw.get_time()-self.start_time)/5*math.pi - math.pi/2)/2+0.5

		if self.mode != self.postssmode+13:
			self.cart.render(self.view, self.projection, self.model, self.transformation, alpha)
			self.pc3.render(self.view, self.projection, self.object, self.transformation, lines=True)
		if len(self.pc2.points) == 0:
			if len(self.pc3.points) == 0:
				self.pc.render(self.view, self.projection, self.object, self.transformation)
		else:
			self.pc2.render(self.view, self.projection, self.object, self.transformation)
		self.line.render(np.identity(4), np.identity(4), np.identity(4), np.identity(4), True)

		if self.mode != self.postssmode+13:
			self.cartesianGuide.render(self.view, self.projection, self.model, self.transformation, True)

		if self.targetModel is not None:
			self.targetModel.render(self.view, self.projection, self.model, self.transformation)

	def linkCallbackFunctions(self, window):
		glfw.set_mouse_button_callback(window, self.onMouseButton)
		glfw.set_key_callback(window, self.inputEvent)
		glfw.set_scroll_callback(window, self.onScroll)

	def onMouseButton(self, window, button, action, mods):
		if button == glfw.MOUSE_BUTTON_LEFT:
			if action == glfw.PRESS:
				print(glfw.get_cursor_pos(window))

	def onScroll(self, window, xOffset, yOffset):
		if self.mode == 1:
			self.model_y_rot += yOffset
		elif self.mode == 3:
			self.model_x_rot += yOffset
		elif self.mode == 5:
			self.model_z_rot += yOffset
		elif self.mode == 7:
			self.model_z_bounds[0] += yOffset/30
		elif self.mode == 8:
			self.model_z_bounds[1] += yOffset/30
		elif self.mode == 10:
			self.model_x_bounds[0] += yOffset/30
		elif self.mode == 11:
			self.model_x_bounds[1] += yOffset/30
		elif self.mode == 13:
			self.model_y_bounds[0] += yOffset/30
		elif self.mode == 14:
			self.model_y_bounds[1] += yOffset/30
		else:
			self.xScroll -= xOffset
			self.yScroll += yOffset

	def inputEvent(self, window, key, scancode, action, mods):
		if action == glfw.PRESS:
			if key == glfw.KEY_ESCAPE: # Creature comfort, escape safely terminates the program
				glfw.set_window_should_close(window, GL_TRUE)
			if key == glfw.KEY_SPACE and self.mode == self.postssmode+13:
				self.cameraIndex += 1
				if self.cameraIndex == len(self.cameras):
					self.cameraIndex = 1
					return
				image_points, name = utility.backprojectPoints2(self.pc3.points, self.cameras[self.cameraIndex])
				self.cameraTransform = utility.transformationFromCamera(self.cameras[self.cameraIndex])
				image = np.array(utility.loadImage("EngineResources/colmapData/drill_full/"+name))
				imps = image_points.reshape(-1,6)
				for pi in range(len(imps)):
					imp = imps[pi]
					x = imp[0]
					y = -imp[1]
					x += 1
					x = x/2
					y += 1
					y = y/2
					x = int(x*1920)
					y = int(y*1080)
					x = np.min([x, 1920-1])
					x = np.max([x, 0])
					y = np.min([y, 1080-1])
					y = np.max([y, 0])
					color = image[y][x]
					self.oldpoints[pi*6+3] = color[0]/255
					self.oldpoints[pi*6+4] = color[1]/255
					self.oldpoints[pi*6+5] = color[2]/255
				self.targetModel = TargetModel(self.oldpoints, self.rotations, self.placementMax)

			if key == glfw.KEY_ENTER and self.mode != self.postssmode+13:
				self.mode += 1
