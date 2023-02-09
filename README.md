# points2pano
LiDAR point clouds to 360 panorama, **Feb 2023**

inealey@ucsd.edu

The program projects a set of 3D scene points (scanned with LiDAR) to the unit sphere for visualization.

Panoramic images can provide an immersive view of a complex scanned scene without access to a full-resolution point cloud.
Additionally, the point density, and therefore view quality, from a single Terrestrial LiDAR Scan (TLS) is highly position dependent.

### Setup:
Requires Python >= 3.8
```
git clone https://github.com/inealey/points2pano.git
cd points2pano
pip install -r requirements.txt
```
  
### Demo:
To run the example:
```
python3 points2pano.py -s 10
```

This will produce an image from the included example point cloud located at `examples/example.las`. 
the `-s` flag increases the default point size a bit because the example is intentionally sparse, ~300,000 points.
That should produce an image like this:
![Sparse Demo](example/demo_subsampled.jpg)

A full-resolution TLS scan (this one has 7.5M points) results in an image like this:
![Fullres Demo](example/demo_fullres.jpg)
 
These image files may be opened with any 360 panorama viewer. Many standalone, web-based, and mobile-based applications are out there.
 
### Usage:
```
usage: points2pano [-h] [-i INPUT] [-o OUTPUT] [-n THREADS] [-y HEIGHT] [-s SIZE] [--skybox] [--no-skybox]

project point clouds to 360 photo sphere

optional arguments:
  -h, --help            show this help message and exit
  -i INPUT, --input INPUT
                        path to .las/.laz LiDAR point cloud file
  -o OUTPUT, --output OUTPUT
                        name of output image file
  -n THREADS, --threads THREADS
                        number of threads
  -y HEIGHT, --height HEIGHT
                        size (height) of the output image. width = 2 * height
  -s SIZE, --size SIZE  point size multiplier
  --skybox              draw sky backgound behind point cloud
  --no-skybox           don't include the skybox
  ```
  
### Performance:
The processing speed is mostly dependent on the number of points in the input file.
Thanks to numpy and vector math, the performance is reasonable, although the main focus here is still readbility while we experiment with features.
 
On my home workstation, I generally observe a processing rate of between 150,000 and 200,000 points per second.
When generating the images above, the subsampled point cloud (278,981 points) took ~8 seconds and the full resolution cloud (7,589,960 points) took ~55 seconds on a single thread.
Leveraging all 8 cores on my machine, I can project the "full resolution" 7.5M point cloud in ~18 seconds. 
 
### TO DO:
- still some issues on Windows due to the shared memory implementation.
