# https://codeloop.org/python-modern-opengl-perspective-projection/
# https://stackoverflow.com/questions/38398699/opengl-python-display-an-image

import glfw
from OpenGL.GL import *
import OpenGL.GL.shaders
import numpy as np
import pyrr
from PIL import Image
import math
from engine_settings import *
import shaders
import utility
from scenes import *

def engineLoop():

	if not glfw.init(): # If GLFW fails to initialize, terminate program
		return

	monitors = glfw.get_monitors()
	active_monitor = monitors[0] # Default to first monitor

	display = [DISPLAY_WIDTH, DISPLAY_HEIGHT] # Target display resolution

	# glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
	# glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)

	window = glfw.create_window(display[0], display[1], "CMPUT428 Graphics Engine", active_monitor, None)
 
	if not window: # If window failed to create, terminate program
		glfw.terminate()
		return
 
	glfw.make_context_current(window) # Set GLFW context to selected window

	activeScene = IntroScene()
	# activeScene = CubeScene()
	activeScene.linkCallbackFunctions(window)

	last_time_seconds = math.floor(glfw.get_time())
	time_seconds = last_time_seconds
	frames = 0
	# Application loop
	while not glfw.window_should_close(window):
		# frames += 1

		# if last_time_seconds != time_seconds and frames != 0:
		# 	print("FPS:" + " "*(math.ceil(frames/10)) + str(frames))
		# 	last_time_seconds = time_seconds
		# 	frames = 0


		# Poll events to trigger any input events
		glfw.poll_events()

		activeScene.sceneLoop(window)

		if activeScene.shouldEnd == True:
			activeScene = CubeScene()
			activeScene.linkCallbackFunctions(window)
		
		# Set rendered texture to window
		glfw.swap_buffers(window)

		time_seconds = math.floor(glfw.get_time())

	# When application loop ends, terminate program
	glfw.terminate()

if __name__ == '__main__':
	engineLoop()
	#theMatrix(1)