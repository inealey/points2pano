### POINTS2PANO ###
### Isaac Nealey 2023 ###
### project a LiDAR point cloud to a sphere to see panorama
### assumes 64-bit floating-point coordinates
### and 8-bit integer RGB channels

import laspy
import numpy as np
from math import floor, sqrt, atan2, acos, pi
import cv2
import argparse
from tqdm import tqdm
from multiprocessing import Process, shared_memory, cpu_count

# constants
NUM_WORKERS = cpu_count()
CANVAS_ARRAY_NAME = 'npsharedcanvas'
SKYBOX_PATH = 'skybox/skybox2.jpg'
PRECISION = 12 ## how far to bitshift when plotting pts

## create shared memory block from numpy-like object
## TODO: still some issues with shm on windows
def create_shared_memory_nparray(data, name, dtype):
    d_size = np.dtype(dtype).itemsize * np.prod(data.shape)
    shm = shared_memory.SharedMemory(create=True, size=d_size, name=name)
    dst = np.ndarray(shape=data.shape, dtype=dtype, buffer=shm.buf)
    dst[:] = data[:]
    return shm

## free a shared memory block
def release_shared(name):
    shm = shared_memory.SharedMemory(name=name)
    shm.close()
    shm.unlink()

## project an set of points to the unit sphere
## return spherical coordinates and
## preserve intensity and color info
## inputs: 
## las: laspy.lasdata.LasData object - from laspy.read()
## canvas_shape: canvas dimensions
## pts_size: point size multiplier
## r_min: minimum aboslute distance
## r_max: maximum absolute distance
## img_type: string to decide how to color points
## start: index into arrays, starting point for reading/writing
## length: how far into the arrays to read/write
def projectPoints(las, canvas_shape, pts_size, r_min, r_max, img_type, start, length):
    
    ## access the shared arrays
    shm_output = shared_memory.SharedMemory(name=CANVAS_ARRAY_NAME)
    image = np.ndarray(canvas_shape, dtype=np.uint8, buffer=shm_output.buf)
    factor = 2 ** PRECISION
    
    ## project each point
    for i in tqdm(range(start, start+length)):
        r = sqrt(las.x[i] ** 2 + las.y[i] ** 2 + las.z[i] ** 2) # absolute distance
        r_norm = (r - r_min) / (r_max - r_min) ## normalized distance. 0=close 1=far
        radius = int(factor * pts_size * (1 - r_norm) + 1) ## linear falloff TODO better nonlinear function? seems ok for 4k
        
        if img_type == 'BGR': ## standard BGR
            color = (int(las.blue[i]), int(las.green[i]), int(las.red[i]))
        elif img_type == 'intensity': ## as an intensity image
            color = (int(las.intensity[i] * 255), int(las.intensity[i] * 255), int(las.intensity[i] * 255))
        elif img_type == 'distance': ## as a distance image
            color = (int(r_norm * 255), int(r_norm * 255), int(r_norm * 255))
        
        ## draw pts with subpixel coordinates
        cv2.circle(image,
            (int(round(((atan2(las.y[i], las.x[i]) + pi) /  (2 * pi) * image.shape[1]) * factor)), # x coordinate theta
            int(round((acos(las.z[i] / r) / pi * image.shape[0]) * factor))), # y coordinate phi
            radius = radius,
            color = color,
            thickness = cv2.FILLED, ## fill in circles
            shift = PRECISION ) ## shift over
        

if __name__ == '__main__':
    ## set up argparser
    parser = argparse.ArgumentParser(
        prog = 'points2pano',
        description = 'project point clouds to 360 photo sphere' )
    parser.add_argument('-i', '--input', type=str, default='example.las', help='path to .las/.laz LiDAR point cloud file')
    parser.add_argument('-o', '--output', type=str, default='output.png', help='name of output image file')
    parser.add_argument('-n', '--threads', type=int, default=1, help='number of threads')
    parser.add_argument('-y', '--height', type=int, default=3840, help='size (height) of the output image. width = 2 * height')
    parser.add_argument('-s', '--size', type=int, default=2, help='point radius in pixels')
    parser.add_argument('-t', '--type', type=str, default='BGR', help='image type.')
    parser.add_argument('--skybox', action='store_true', default=True)
    parser.add_argument('--no-skybox', dest='skybox', action='store_false')
    args = parser.parse_args()
    
    if args.threads > NUM_WORKERS: exit('too many threads requested')
    if args.type != 'BGR' and args.type != 'distance' and args.type != 'intensity':
        exit('did not recognize requested image type')
        
    ## read point cloud
    las = laspy.read(args.input)
    
    ## convenience vars
    out_shape = (args.height, args.height*2, 3)
    procs = [] ## process list
    height = args.height
    width = args.height * 2
    
    ## compute and normalize distance vector
    r = np.sqrt(np.array(las.x) ** 2 + np.array(las.y) ** 2 + np.array(las.z) ** 2) ## absolute distance
    r_min = min(r)
    r_max = max(r)
    del r
    
    ## skybox or black background.
    if not args.skybox:
        background = np.zeros([height, width, 3])
    else:
        try:
            background = cv2.imread(SKYBOX_PATH)
            background = np.array(cv2.resize(background, (width, height)))
        except cv2.error as e:
            print(e)
            background = np.zeros([height, width, 3])
    
    ## allocate shared memory for output array and distance vector
    shm_output = create_shared_memory_nparray(background, CANVAS_ARRAY_NAME, np.uint8)
    
    ## project to spherical coordinates
    ## compute start and length for indexing points array 
    for i in range(args.threads):
        start = floor(i * las.header.point_count / args.threads)
        length = floor(las.header.point_count * (i + 1) / args.threads) - floor(las.header.point_count * i / args.threads)
        p = Process(target = projectPoints, args = (las, out_shape, args.size, r_min, r_max, args.type, start, length))
        procs.append(p)
        p.start()
    
    # complete the projection processes
    for proc in procs:
        proc.join()
    
    ## access the shared memory
    image = np.ndarray(out_shape, dtype=np.uint8, buffer=shm_output.buf)
    
    ## save image
    cv2.imwrite(args.output, image)
    
    ## clean up shm
    release_shared(CANVAS_ARRAY_NAME)
    
## REFERENCE:
## https://stackoverflow.com/questions/15658145/how-to-share-work-roughly-evenly-between-processes-in-mpi-despite-the-array-size
## from: https://luis-sena.medium.com/sharing-big-numpy-arrays-across-python-processes-abf0dc2a0ab2
## https://github.com/PacktPublishing/Mastering-OpenCV-4-with-Python/blob/0e136cb1771a0c96da67009f623e5840154d55c8/Chapter04/01-chapter-content/shift_parameter.py#L23-L29
## http://moodware.cz/project/city-street-skyboxes-vol-1/