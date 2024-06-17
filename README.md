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

- You will need conventional RGB projector, and a RGB camera with a diffraction grating infront of the projector.

- Calibration between camera-projector and camera-diffraction grating must be done in advance.


## Calibration datasets

![image](https://github.com/shshin1210/DSL/assets/80568500/2061bd98-1b70-4526-a0a8-f324585c2a6d)

You should have a complete form of calibrated parpameters that fits to your imaging system configureation.

- Camera & projector intrinsic and extrinsic paramters

- Camera response function & projector emission function & Diffraction grating efficiency

- First-order corresponding model

We provide an expample calibration parameters in our [DSL Calibration Parameters](https://drive.google.com/drive/folders/18HVXZuSfRsm4V31oBXS9DjdMzSNVSjcO?usp=sharing).


## Datasets
You need to prepare three types of datasets for hyperspectral reconstruction using our DSL method.

Here are some steps to capture a single scene. Please refer to [DSL Supplementary](https://arxiv.org/pdf/2311.18287) for more details and we also provide example of captured datasets in our [DSL Calibration Parameters](https://drive.google.com/drive/folders/18HVXZuSfRsm4V31oBXS9DjdMzSNVSjcO?usp=sharing) named tedey_bear_datasets.zip.

1. Scene's depth map

   https://github.com/shshin1210/DSL/assets/80568500/52a04828-5dad-4c4d-9d49-382ad86a81db

   - Capture a scene under binary code pattern with a specific exposure time where zoer-order light is valid and first-order dispersion intensity is invalid

   - By utilizing conventional structured light decoding method, you should be able to prepare depth reconstructed result. Save the depth result as npy file.
      
2. Scene under white scan line pattern

   https://github.com/shshin1210/DSL/assets/80568500/c4c52964-c5c3-4915-a6ee-606ef3420bf6
   
   - Capture the scene under white scan line pattern with two different intensity pattern values and exposure time.
   
   - Save it in `path_to_ldr_exp1`, `path_to_ldr_exp2`.

4. Scene under black pattern and white pattern
   
   We need scene captured under black pattern with two different intensity pattern values and exposure time same as step 2.

   - Save it in `path_to_black_exp1`, `path_to_black_exp2`.
   
   Also, capture the scene under white pattern under two different intensity pattern values to calculate the radiance weight (normalization) for two different settings same as step 2.

   - Save it in `path_to_intensity1`, `path_to_intensity2`.

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

## Hyperspectral Reconstruction
If you have prepared all datasets, start reconstructing hyperspectral reflectance:
```
python hyper_sl/hyperspectral_reconstruction.py
```

replace any configuration changes in [ArgParse.py] file (https://github.com/shshin1210/DSL/blob/main/hyper_sl/utils/ArgParser.py).

