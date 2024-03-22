import argparse
import torch, os

class Argument:
	def __init__(self):
		self.parser = argparse.ArgumentParser()

		self.parser.add_argument('--device', type = str, default="cuda:0")

		# PATH
		self.parser.add_argument('--calibration_param_path', type = str, default="./calibration/calibration_propcam.xml")

		self.parser.add_argument('--illum_dir', type = str, default="./dataset/image_formation/illum/line_pattern_5") 
		self.parser.add_argument('--dat_dir', type = str, default='./dataset/image_formation/dat') 
  
		self.parser.add_argument('--dg_intensity_dir', type=str, default='./dataset/image_formation/DG.npy')
		self.parser.add_argument('--pef_dir', type = str, default="./dataset/image_formation/PEF.npy") 
		self.parser.add_argument('--crf_dir', type = str, default="./dataset/image_formation/CRF.npy") 

		self.parser.add_argument('--real_data_dir', type=str, default="./dataset/data")
		self.parser.add_argument('--position_calibrated_data_dir', type = str, default="./dataset/image_formation/2023%s/npy_data")
  
		self.parser.add_argument('--real_data_date', type = str, default="1003")
		self.parser.add_argument('--calibrated_date', type =str, default="1007")
  
  
  
		# for HDR imaging
		self.parser.add_argument('--exp_min', help= 'minimum exposure', type = int, default=160)
		self.parser.add_argument('--exp_max', help= 'maximum exposure', type = int, default=320)
  
		self.parser.add_argument('--intensity_min', help= 'minimum exposure', type=float, default=0.2)
		self.parser.add_argument('--intensity_max', help= 'maximum exposure', type=float, default=0.8)

		self.parser.add_argument('--intensity_normalization_pts', help= 'maximum exposure', type=list, default=[128, 517])

		self.parser.add_argument('--invalid_intensity_ratio', type = float, default= 0.01)
    
    
    
    
    
		self.parser.add_argument('--wvl_min', type = float, default= 430e-9) # 430e-9
		self.parser.add_argument('--wvl_max', type = float, default= 660e-9) # 660e-9
		self.parser.add_argument('--wvl_num', type = int, default= 24) # 24 개 (for 10nm interval) 47개 (for 5nm interval)
		self.parser.add_argument('--new_wvl_num', type = int, default=47)
  
		self.parser.add_argument('--depth_min', type = float, default= 600*1e-3) # n단위
		self.parser.add_argument('--depth_max', type = float, default= 900*1e-3) 
		self.parser.add_argument('--depth_interval', type = int, default= 0.001) # 1mm 단위 : 0.001
		
		self.parser.add_argument('--m_min', type = int, default= -1) 
		self.parser.add_argument('--m_max', type = int, default= 1) 
		self.parser.add_argument('--m_num', type = int, default=3)

		self.parser.add_argument('--epoch_num', type = int, default= 2000)
		self.parser.add_argument('--cam_W', type = int, default= 890)
		self.parser.add_argument('--cam_H', type = int, default= 580)
  
		self.parser.add_argument('--proj_W', type = int, default= 1280//2) # 640
		self.parser.add_argument('--proj_H', type = int, default= 720//2) # 360
  
		################## TODO: clean the following codes

		self.parser.add_argument('--focal_length', type=float, default= 12) # 16mm
		
  		# for Camera calibaration
		self.parser.add_argument('--sensor_width', type=float, default=5.32) # 7.18mm
		self.parser.add_argument('--sensor_height', type=float, default=5.32) # 5.32mm
		self.parser.add_argument('--baseline', type=float, default=0.1) # 10cm
		self.parser.add_argument('--cam_pitch', type=float, default=3.45*1e-6*2) # 3.45um *2 (binning)

		# projector
		self.parser.add_argument('--sensor_diag_proj',  type = float, default= 5.842) # 5.842mm
		
	def parse(self):
		arg = self.parser.parse_args()
  
		arg.wvls = torch.linspace(arg.wvl_min, arg.wvl_max, arg.wvl_num)
		arg.new_wvls = torch.linspace(arg.wvl_min, arg.wvl_max, arg.new_wvl_num)
		arg.m_list = torch.linspace(arg.m_min, arg.m_max, arg.m_num)
		arg.depth_list = torch.arange(arg.depth_min, arg.depth_max, arg.depth_interval)
		arg.illum_num = len(os.listdir(arg.illum_dir))
		arg.sensor_height_proj = (torch.sin(torch.atan2(torch.tensor(arg.proj_H),torch.tensor(arg.proj_W)))*(arg.sensor_diag_proj*1e-3))
		arg.proj_pitch = (arg.sensor_height_proj/ (arg.proj_H))

		return arg