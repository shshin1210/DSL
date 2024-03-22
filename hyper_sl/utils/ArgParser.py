import argparse
import torch, os

class Argument:
	def __init__(self):
		self.parser = argparse.ArgumentParser()

		self.parser.add_argument('--device', type = str, default="cuda:0")

		# PATH
		self.parser.add_argument('--calibration_param_path', help='calibration parameter xml file path', type = str, default="./calibration/calibration_propcam.xml")
		self.parser.add_argument('--illum_dir', help='path to white scan line illumination pattern directory', type = str, default="./dataset/image_formation/illum/line_pattern_5") 
  
		self.parser.add_argument('--dg_intensity_dir', help='path to diffraction grating efficiency directory', type=str, default='./dataset/image_formation/DG.npy')
		self.parser.add_argument('--pef_dir', help='path to projector emission function directory', type = str, default="./dataset/image_formation/PEF.npy") 
		self.parser.add_argument('--crf_dir', help='path to camera response function directory', type = str, default="./dataset/image_formation/CRF.npy") 

		self.parser.add_argument('--real_data_dir', help='path to real dataset directory', type=str, default="./dataset/data/realdata")
		self.parser.add_argument('--position_calibrated_data_dir', help='directory of calibrated data driven correspondence model', type = str, default="./dataset/image_formation/first_order_correspondence_model.npy")

		# PATH for HDR imaging
		self.parser.add_argument('--path_to_intensity1', help='path to white pattern captured image png for intensity1', type = str, default="./dataset/data/intensity1/*.png")
		self.parser.add_argument('--path_to_intensity2', help='path to white pattern captured image png for intensity2', type = str, default="./dataset/data/intensity2/*.png")
  
		self.parser.add_argument('--path_to_black_exp1', help='path to black pattern captured image png under exposure1', type = str, default="./dataset/data/black_exposure1/*.png")
		self.parser.add_argument('--path_to_black_exp2', help='path to black pattern captured image png under exposure2', type = str, default="./dataset/data/black_exposure2/*.png")

		self.parser.add_argument('--path_to_ldr_exp1', help='path to ldr images under exposure1', type = str, default="./dataset/data/ldr_exposure1")
		self.parser.add_argument('--path_to_ldr_exp2', help='path to ldr images under exposure1', type = str, default="./dataset/data/ldr_exposure2")

		# PATH for hyperspectral reconstruction
		self.parser.add_argument('--path_to_depth', help='path to depth npy data for specific scene', type = str, default="./dataset/data/depth.npy")
		self.parser.add_argument('--path_to_intensity1', help='path to white pattern captured image png for intensity1', type = str, default="./dataset/data/intensity1/*.png")

		# for HDR imaging
		self.parser.add_argument('--exp_min', help= 'minimum exposure', type = int, default=160)
		self.parser.add_argument('--exp_max', help= 'maximum exposure', type = int, default=320)
  
		self.parser.add_argument('--intensity_min', help= 'minimum intensity of pattern', type=float, default=0.2)
		self.parser.add_argument('--intensity_max', help= 'maximum intensity of pattern', type=float, default=0.8)

		self.parser.add_argument('--intensity_normalization_pts', help= 'x, y location of color checker black patch', type=list, default=[128, 517])
		self.parser.add_argument('--invalid_intensity_ratio', help= 'threshold for invalid intensity', type = float, default= 0.01)
  
		self.parser.add_argument('--max_intensity', help='maximum intensity of a image (default uint16)', type=int, default=2**16)
    
		# wavelengths
		self.parser.add_argument('--wvl_min', help='minimum wavelength', type = float, default= 430e-9) # 430e-9
		self.parser.add_argument('--wvl_max', help='maximum wavelength', type = float, default= 660e-9) # 660e-9
		self.parser.add_argument('--wvl_num', help='maximum intensity of a image', type = int, default= 24) # 24 개 (for 10nm interval) 47개 (for 5nm interval)
		self.parser.add_argument('--new_wvl_num', help='number of new wavelength range', type = int, default=47)

		# depths
		self.parser.add_argument('--depth_min', help='minimum depth', type = float, default= 600*1e-3) # n단위
		self.parser.add_argument('--depth_max',  help='maximum depth', type = float, default= 900*1e-3) 
		
		# diffraction grating m orders
		self.parser.add_argument('--m_min', help='minimum m order', type = int, default= -1) 
		self.parser.add_argument('--m_max', help='maximum m order', type = int, default= 1) 
		self.parser.add_argument('--m_num', help='number of m orders', type = int, default=3)

		# camera parameters
		self.parser.add_argument('--cam_W', help ='width of camera', type = int, default= 890)
		self.parser.add_argument('--cam_H', help ='height of camera', type = int, default= 580)

		# Mask
		self.parser.add_argument('--sigma', help ='Mask blur sigma value', type = int, default= 150)

		# optimization
		self.parser.add_argument('--lr', help='learning rate for optimization', type = float, default= 0.04)
		self.parser.add_argument('--decay_step', help='decay step for optimization', type = int, default= 500)
		self.parser.add_argument('--gamma', help='gamma value for optimization', type = float, default= 0.5)
		self.parser.add_argument('--epoch_num', help='number of epoch for optimization', type = int, default= 2000)
		self.parser.add_argument('--depth_scalar', help='depth scalar', type = float, default= 4.5*1e5)
		self.parser.add_argument('--weight_first', help='weight for first order', type = int, default= 1)
		self.parser.add_argument('--weight_zero', help='weight for zero order', type = int, default= 0)
		self.parser.add_argument('--weight_unreach', help='weight or unreached wavelengths', type = int, default= 1)
		self.parser.add_argument('--weight_spectral', help='weight for spectral smoothness', type = int, default= 50)

	def parse(self):
		arg = self.parser.parse_args()
  
		arg.wvls = torch.linspace(arg.wvl_min, arg.wvl_max, arg.wvl_num)
		arg.new_wvls = torch.linspace(arg.wvl_min, arg.wvl_max, arg.new_wvl_num)
		arg.m_list = torch.linspace(arg.m_min, arg.m_max, arg.m_num)
		arg.depth_list = torch.arange(arg.depth_min, arg.depth_max, arg.depth_interval)
		arg.illum_num = len(os.listdir(arg.illum_dir))

		return arg