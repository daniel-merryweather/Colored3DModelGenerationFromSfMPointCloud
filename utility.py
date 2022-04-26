from PIL import Image
import numpy as np
import pyrr
import json
import struct
import scipy.spatial

""" NOT MY CODE !!!!!!!!!!!!!!!!!!! """


# Copyright (c) 2022, ETH Zurich and UNC Chapel Hill.
# All rights reserved.

def read_next_bytes(fid, num_bytes, format_char_sequence, endian_character="<"):
    """Read and unpack the next bytes from a binary file.
    :param fid:
    :param num_bytes: Sum of combination of {2, 4, 8}, e.g. 2, 6, 16, 30, etc.
    :param format_char_sequence: List of {c, e, f, d, h, H, i, I, l, L, q, Q}.
    :param endian_character: Any of {@, =, <, >, !}
    :return: Tuple of read and unpacked values.
    """
    data = fid.read(num_bytes)
    return struct.unpack(endian_character + format_char_sequence, data)

def read_points3D_binary(path_to_model_file):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadPoints3DBinary(const std::string& path)
        void Reconstruction::WritePoints3DBinary(const std::string& path)
    """
    points3D = {}
    with open(path_to_model_file, "rb") as fid:
        num_points = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_points):
            binary_point_line_properties = read_next_bytes(
                fid, num_bytes=43, format_char_sequence="QdddBBBd")
            point3D_id = binary_point_line_properties[0]
            xyz = np.array(binary_point_line_properties[1:4])
            rgb = np.array(binary_point_line_properties[4:7])
            error = np.array(binary_point_line_properties[7])
            track_length = read_next_bytes(
                fid, num_bytes=8, format_char_sequence="Q")[0]
            track_elems = read_next_bytes(
                fid, num_bytes=8*track_length,
                format_char_sequence="ii"*track_length)
            image_ids = np.array(tuple(map(int, track_elems[0::2])))
            point2D_idxs = np.array(tuple(map(int, track_elems[1::2])))
            points3D[point3D_id] = [xyz, rgb]
    return points3D

def read_images_binary(path_to_model_file):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadImagesBinary(const std::string& path)
        void Reconstruction::WriteImagesBinary(const std::string& path)
    """
    images = {}
    with open(path_to_model_file, "rb") as fid:
        num_reg_images = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_reg_images):
            binary_image_properties = read_next_bytes(
                fid, num_bytes=64, format_char_sequence="idddddddi")
            image_id = binary_image_properties[0]
            qvec = np.array(binary_image_properties[1:5])
            tvec = np.array(binary_image_properties[5:8])
            camera_id = binary_image_properties[8]
            image_name = ""
            current_char = read_next_bytes(fid, 1, "c")[0]
            while current_char != b"\x00":   # look for the ASCII 0 entry
                image_name += current_char.decode("utf-8")
                current_char = read_next_bytes(fid, 1, "c")[0]
            num_points2D = read_next_bytes(fid, num_bytes=8,
                                           format_char_sequence="Q")[0]
            x_y_id_s = read_next_bytes(fid, num_bytes=24*num_points2D,
                                       format_char_sequence="ddq"*num_points2D)
            xys = np.column_stack([tuple(map(float, x_y_id_s[0::3])),
                                   tuple(map(float, x_y_id_s[1::3]))])
            point3D_ids = np.array(tuple(map(int, x_y_id_s[2::3])))
            images[image_id] = [qvec, tvec, image_name]
    return images

def qvec2rotmat(qvec):
    return np.array([
        [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]])




""" Okay now everything else is my own, no more COLMAP copyright """

def reverseTransformations(points, transformations):
	points = points.copy()
	for tf in transformations[::-1]:
		R = tf[0:3,0:3]
		t = tf[0:3,3]
		reverse = np.identity(4)
		reverse[0:3,0:3] = R.T
		reverse[0:3,3] = -t
		points = transformPoints(points, reverse)
	return points

def backprojectPoints(points3d, camera):
	rot_quat = camera[0]

	translation = camera[1]

	name = camera[2]

	rot_mat = qvec2rotmat(rot_quat)

	f = 1680

	intrinsic = np.identity(3)
	intrinsic[0,0] = f
	intrinsic[1,1] = f

	extrinsic = np.zeros([3,4])
	extrinsic[0:3,0:3] = rot_mat
	extrinsic[0:3,3] = translation

	camera_mat = intrinsic.dot(extrinsic)

	points3d = np.array(list(points3d.values()))
	points = points3d[:,0]

	homogeneous_points = np.array([points[:,0],points[:,1],points[:,2],[1]*len(points)])

	image_points = camera_mat.dot(homogeneous_points)
	image_points[0] = image_points[0] / image_points[2]
	image_points[1] = image_points[1] / image_points[2]
	image_points[2] = image_points[2] / image_points[2]

	new_image_points = np.zeros([len(image_points[0]),6])
	new_image_points[:,0] = image_points[0]/1920*2# + 0.5
	new_image_points[:,1] = -image_points[1]/1080*2# - 0.5
	new_image_points[:,2] = [0]*len(image_points[2])
	new_image_points[:,3] = points3d[:,1][:,0]/255
	new_image_points[:,4] = points3d[:,1][:,1]/255
	new_image_points[:,5] = points3d[:,1][:,2]/255

	new_image_points = new_image_points.flatten()

	return new_image_points, name

def transformationFromCamera(camera):
	rot_quat = camera[0]

	translation = camera[1]

	rot_mat = qvec2rotmat(rot_quat)

	extrinsic = np.identity(4)
	extrinsic[0:3,0:3] = -rot_mat.T
	#extrinsic[0:3,3] = translation

	return extrinsic

def backprojectPoints2(points, camera):
	rot_quat = camera[0]

	translation = camera[1]

	name = camera[2]

	rot_mat = qvec2rotmat(rot_quat)

	f = 1680

	intrinsic = np.identity(3)
	intrinsic[0,0] = f
	intrinsic[1,1] = f

	extrinsic = np.zeros([3,4])
	extrinsic[0:3,0:3] = rot_mat
	extrinsic[0:3,3] = translation

	camera_mat = intrinsic.dot(extrinsic)

	homogeneous_points = np.array([points[0::6],points[1::6],points[2::6],[1]*len(points[0::6])])

	image_points = camera_mat.dot(homogeneous_points)
	image_points[0] = image_points[0] / image_points[2]
	image_points[1] = image_points[1] / image_points[2]
	image_points[2] = image_points[2] / image_points[2]

	new_image_points = np.zeros([len(image_points[0]),6])
	new_image_points[:,0] = image_points[0]/1920*2# + 0.5
	new_image_points[:,1] = -image_points[1]/1080*2# - 0.5
	new_image_points[:,2] = [0]*len(image_points[2])
	new_image_points[:,3] = points[3::6]
	new_image_points[:,4] = points[4::6]
	new_image_points[:,5] = points[5::6]

	new_image_points = new_image_points.flatten()

	return new_image_points, name

def loadJSON(path):
	data = None
	with open(path, "r") as f:
		data = json.load(f)
	return data

def loadCameraAngle(path):
	data = loadJSON(path)
	return data["camera_angle_x"], data["camera_angle_y"]

def loadImageData(path):
	image = Image.open(path)
	image_data = np.array(list(image.getdata()), np.uint8)
	return image, image_data

def loadImage(path):
	image = Image.open(path)
	return image

def loadJSONCameraParams(path):
	file = open(path)
	data = json.loads(file.read())
	matrix = data['extrinsic']
	matrix = np.array(matrix)
	c_x, c_y = data['c_x'], data['c_y']
	f_x, f_y = data['f_x'], data['f_y']
	return c_x, c_y, f_x, f_y, matrix

def heuristicDensity(points, point, minimum):
	points = points.T
	distances = scipy.spatial.distance.cdist(points, [point])
	return np.count_nonzero(distances < minimum)

def createTransformationMatrix(xPos=0, yPos=0, zPos=0, xRotation=0, yRotation=0, zRotation=0):
	rot_x = pyrr.Matrix44.from_x_rotation(-xRotation)
	rot_y = pyrr.Matrix44.from_y_rotation(yRotation)
	rot_z = pyrr.Matrix44.from_z_rotation(zRotation)

	transformationMatrix = rot_z @ rot_y @ rot_x # Combine rotation matricies

	transformationMatrix[3][0] = xPos # Set translation for x, y and z
	transformationMatrix[3][1] = yPos
	transformationMatrix[3][2] = zPos

	return transformationMatrix

def transformPoints(points, transform, scalar=1):
	newPoints = np.zeros(len(points))

	x = points[0::6]
	y = points[1::6]
	z = points[2::6]

	r = points[3::6]
	g = points[4::6]
	b = points[5::6]

	p = np.array([x,y,z,[1]*len(x)])

	p = transform.dot(p)

	newPoints[0::6] = p[0]
	newPoints[1::6] = p[1]
	newPoints[2::6] = p[2]

	newPoints[3::6] = r
	newPoints[4::6] = g
	newPoints[5::6] = b

	return newPoints

def removeZeroPoints(points):
	newsize = len(points)
	for i in range(int(len(points)/6)):
		if points[i*6+0] == 0 and points[i*6+1] == 0 and points[i*6+2] == 0:
			newsize -= 6
	newPoints = np.zeros(newsize)
	index = 0
	for i in range(int(len(points)/6)):
		if points[i*6+0] == 0 and points[i*6+1] == 0 and points[i*6+2] == 0:
			continue
		newPoints[index*6+0] = points[i*6+0]
		newPoints[index*6+1] = points[i*6+1]
		newPoints[index*6+2] = points[i*6+2]
		newPoints[index*6+3] = points[i*6+3]
		newPoints[index*6+4] = points[i*6+4]
		newPoints[index*6+5] = points[i*6+5]
		index += 1
	return newPoints

def getCameraTransformation(data, frame):
	for f in data["frames"]:
		name = f["file_path"][-9:-4]
		if name == frame:
			transform = np.array(f["transform_matrix"])
			rearrange = np.array([
				[1,0,0,0],
				[0,0,1,0],
				[0,1,0,0],
				[0,0,0,1]])
			transform = rearrange.dot(transform)
			return transform
	return -1

def binaryHash1D(arr, unitSize=0.25):
	lower = np.min(arr)
	upper = np.max(arr)
	size = np.ceil((upper-lower)/unitSize)
	table = []
	for i in range(int(size)+1):
		localRange = []
		localLower = lower + i*unitSize
		localUpper = lower + (i+1)*unitSize
		for vi in range(len(arr)):
			if arr[vi] >= localLower and arr[vi] < localUpper:
				localRange.append(vi)
		table.append(localRange)
	return table, lower

def extractRangeFromBinaryHash(binhash, lower, upper):
	return np.concatenate(np.array(binhash[lower:upper])).ravel()

def binaryHashFrom3DPoints(points, unitSize=0.25):
	x = points[0::6]
	y = points[1::6]
	z = points[2::6]

	xhash, xlower = binaryHash1D(x, unitSize)
	yhash, ylower = binaryHash1D(y, unitSize)
	zhash, zlower = binaryHash1D(z, unitSize)

	return np.array([xhash, yhash, zhash, xlower, ylower, zlower])

def positionCubes(points, cubeIndicies, unitSize):
	xpoints = points[0::6]
	xlower = np.min(xpoints)

	ypoints = points[1::6]
	ylower = np.min(ypoints)

	zpoints = points[2::6]
	zlower = np.min(zpoints)

	positions = []

	for ci in cubeIndicies:
		cxi = ci[0]
		cyi = ci[1]
		czi = ci[2]

		positions.append([xlower + cxi*unitSize, ylower + cyi*unitSize, zlower + czi*unitSize])

	return positions

def extractRangeFrom3DBinaryHash(hashes, xlower, xupper, ylower, yupper, zlower, zupper):
	xindicies = extractRangeFromBinaryHash(hashes[0], xlower, xupper)
	yindicies = extractRangeFromBinaryHash(hashes[1], ylower, yupper)
	zindicies = extractRangeFromBinaryHash(hashes[2], zlower, zupper)

	return np.intersect1d(np.intersect1d(xindicies, yindicies), zindicies).astype('int32')

def filterDensity(points, binhash, minDensity):
	xsize = len(binhash[0])
	ysize = len(binhash[1])
	zsize = len(binhash[2])

	filteredIndicies = []

	for xi in range(xsize):
		for yi in range(ysize):
			for zi in range(zsize):
				indicies = extractRangeFrom3DBinaryHash(binhash, xi, xi+1, yi, yi+1, zi, zi+1)
				if len(indicies) < minDensity:
					for i in indicies:
						filteredIndicies.append([i*6+0, i*6+1, i*6+2, i*6+3, i*6+4, i*6+5])

	if len(filteredIndicies) == 0:
		return points
	points = np.delete(points, np.concatenate(filteredIndicies))

	return points