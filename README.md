# Dispersed Structured Light for Hyperspectral 3D Imaging
[DSL](project page url) (Dispersed Structured Light for Hyperspectral 3D Imaging) is a method that reconstructs high quality depth and hyperspectral information.
You should follow the [requirements.txt](https://github.com/shshin1210/DSL/blob/main/requirements.txt) provided there.

## Installation
```
git clone https://github.com/shshin1210/DSL.git
cd DSL
pip install -r requirements.txt
```

## Datasets
You need to prepare three types of datasets for hyperspectral reconstruction. Refer to [DSL Supplementary](supplementary url) for more details.

1. Scene's depth map
   You should prepare depth reconstructed result using conventional structured light method under binary code patterns.
   Remember this is captured under a specific exposure time where first-order dispersion intensity is invalid.
   
2. Scene under white scan line pattern
   Capture the scene under white scan line pattern with two different intensity pattern values.
   These should be saved in `path_to_ldr_exp1`, `path_to_ldr_exp2`.

3. Scene under black pattern and white pattern
   We need scene captured under black pattern with two different exposure settings.
   Also, capture the scene under white pattern under two different intensity pattern values to calculate the radiance weight (normalization) for two different settings.

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

