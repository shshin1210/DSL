# Dispersed Structured Light for Hyperspectral 3D Imaging
[DSL](https://shshin1210.github.io/DSL/) (Dispersed Structured Light for Hyperspectral 3D Imaging) is a method that reconstructs high quality depth and hyperspectral information.
You should follow the [requirements.txt](https://github.com/shshin1210/DSL/blob/main/requirements.txt) provided there.

## Installation
```
git clone https://github.com/shshin1210/DSL.git
cd DSL
pip install -r requirements.txt
```

## Image system configuration
![image_system](https://github.com/shshin1210/DSL/assets/80568500/d0dc7d9e-d12b-4901-bc9c-91551f896bf1)
Prepare the DSL imaging system configuration as the figure above.
You will need conventional RGB projector, and a RGB camera with a diffraction grating infront of the projector.
Calibration between camera-projector and camera-diffraction grating must be done in advance.

## Datasets
You need to prepare three types of datasets for hyperspectral reconstruction. Refer to [DSL Supplementary](https://arxiv.org/pdf/2311.18287) for more details.

1. Scene's depth map

   You should prepare depth reconstructed result using conventional structured light method under binary code patterns.
   
   Remember this is captured under a specific exposure time where first-order dispersion intensity is invalid.
   
2. Scene under white scan line pattern
   
   Capture the scene under white scan line pattern with two different intensity pattern values.
   
   Save it in `path_to_ldr_exp1`, `path_to_ldr_exp2`.

3. Scene under black pattern and white pattern
   
   We need scene captured under black pattern with two different exposure settings.

   Save it in `path_to_black_exp1`, `path_to_black_exp2`.
   
   Also, capture the scene under white pattern under two different intensity pattern values to calculate the radiance weight (normalization) for two different settings.

   Save it in `path_to_intensity1`, `path_to_intensity2`.

```
dataset
|-- depth.npy
|-- intensity1
|-- intensity2
|-- black_exposure1
|-- black_exposure2
|-- ldr_exposure1
    |-- scene under white scanline pattern 0.png
    |-- scene under white scanline pattern 1.png
    |-- ...
|-- ldr_exposure2
    |-- scene under white scanline pattern 0.png
    |-- scene under white scanline pattern 1.png
    |-- ...
```


## How To Run?
To reconstruct hyperspectral reflectance:
```
python hyper_sl/hyperspectral_reconstruction.py
```

replace any configuration changes in [ArgParse.py] file (https://github.com/shshin1210/DSL/blob/main/hyper_sl/utils/ArgParser.py).

