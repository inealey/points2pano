### POINTS2PANO ###
### Isaac Nealey 2023 ###
### project a LiDAR point cloud to a sphere to see panorama
### assumes 64-bit floating-point coordinates
### and 8-bit integer RGB channels

import laspy
import numpy as np
from math import floor, pi
import cv2
import argparse
from tqdm import tqdm
from multiprocessing import Process, shared_memory, cpu_count

# constants
NUM_WORKERS = cpu_count()
CANVAS_ARRAY_NAME = 'npsharedcanvas'
SKYBOX_PATH = 'skybox/skybox4.jpg'
PRECISION = 16 ## how far to bitshift when plotting pts
## 16 is approx precision of a 64-bit float
FACTOR = 2 ** PRECISION ## bitshift factor (sig figs)

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
## start: index into arrays, starting point for reading/writing
## length: how far into the arrays to read/write
## canvas_shape: canvas dimensions
## theta: array of theta values
## phi: array of phi values
## radius: array of radii
## color: array of BGR colors
def projectPoints(start, length, canvas_shape, theta, phi, radius, color):
    ## access the shared array
    shm_output = shared_memory.SharedMemory(name=CANVAS_ARRAY_NAME)
    image = np.ndarray(canvas_shape, dtype=np.uint8, buffer=shm_output.buf)
    ## project each point,
    ## draw pts with subpixel coordinates
    for i in tqdm(range(start, start+length)):
        cv2.circle(image,
            (theta[i], phi[i]),
            radius = radius[i],
            color = (int(color[i][0]), int(color[i][1]), int(color[i][2])),
            thickness = cv2.FILLED,
            shift = PRECISION )
        
if __name__ == '__main__':
    ## set up argparser
    parser = argparse.ArgumentParser(
        prog = 'points2pano',
        description = 'project point clouds to 360 photo sphere' )
    parser.add_argument('-i', '--input', type=str, default='example.las', 
                help='path to .las/.laz LiDAR point cloud file')
    parser.add_argument('-o', '--output', type=str, default='output.png', 
                help='name of output image file')
    parser.add_argument('-n', '--threads', type=int, default=1, 
                help='number of threads')
    parser.add_argument('-l', '--log', type=str, default='WARNING', 
                help='logging level')
    parser.add_argument('-y', '--height', type=int, default=4320,
                help='size (height) of the output image. width = 2 * height')
    parser.add_argument('-s', '--size', type=int, default=3, 
                help='point radius in pixels')
    parser.add_argument('--skybox', action='store_true', default=True)
    parser.add_argument('--no-skybox', dest='skybox', action='store_false')
    args = parser.parse_args()
    
    if args.threads > NUM_WORKERS: exit('too many threads requested')
        
    ## read point cloud
    las = laspy.read(args.input)
    
    ## convenience vars
    out_shape = (args.height, args.height*2, 3)
    procs = [] ## process list
    height = args.height
    width = args.height * 2
    point_count = las.header.point_count
    
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
            
    ## compute spherical coordinates as vectors
    r = np.sqrt(np.array(las.x) ** 2 + np.array(las.y) ** 2 +
                np.array(las.z) ** 2) ## absolute distances
    
    r_norm = (r - min(r)) / (max(r) - min(r))  ## normalized. 0=close 1=far.
    
    theta = np.array(((np.arctan2(las.y, las.x) + pi) /
                      (2 * pi) * width) * FACTOR, dtype = np.uint64) # x value
    
    phi = np.array((np.arccos(las.z / r) / pi * height) *
                   FACTOR, dtype = np.uint64) # y value
    
    ## radius has linear falloff TODO better nonlinear function?
    radius = np.array(FACTOR * args.size * (1 - r_norm) + 1,
                      dtype = np.uint64) # size of glyph
    
    color = np.swapaxes(np.array((las.blue, las.green, las.red),
                                 dtype = np.uint8), 0, 1) # color of glyph
    
    del las ## explicit free, don't need point cloud anymore
    
    ## allocate shared memory for output array and distance vector
    shm_output = create_shared_memory_nparray(background,
                                              CANVAS_ARRAY_NAME, np.uint8)
    
    ## project to spherical coordinates
    ## compute start and length for indexing points array 
    for i in range(args.threads):
        start = floor(i * point_count / args.threads)
        length = floor(point_count * (i + 1) / args.threads) - \
            floor(point_count * i / args.threads)
        
        p = Process(target = projectPoints,
                args = (start, length, out_shape, theta, phi, radius, color))
        
        procs.append(p)
        p.start()
    
    # complete the projection processes
    for p in procs:
        p.join()
    
    ## access the shared memory
    image = np.ndarray(out_shape, dtype=np.uint8, buffer=shm_output.buf)
    
    ## save image
    try:
        cv2.imwrite(args.output, image)
    except cv2.error as e:
        print(e)
            
    ## clean up shm
    release_shared(CANVAS_ARRAY_NAME)
    
## REFERENCE:
## https://stackoverflow.com/questions/15658145/how-to-share-work-roughly-evenly-between-processes-in-mpi-despite-the-array-size
## from: https://luis-sena.medium.com/sharing-big-numpy-arrays-across-python-processes-abf0dc2a0ab2
## https://github.com/PacktPublishing/Mastering-OpenCV-4-with-Python/blob/0e136cb1771a0c96da67009f623e5840154d55c8/Chapter04/01-chapter-content/shift_parameter.py#L23-L29
## http://moodware.cz/project/city-street-skyboxes-vol-1/