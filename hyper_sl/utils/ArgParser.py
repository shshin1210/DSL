import argparse
import torch

class Argument:
	def __init__(self):
		self.parser = argparse.ArgumentParser()

		self.parser.add_argument('--device', type = str, default="cuda:0")

		################## PATH
		self.parser.add_argument('--output_dir', type = str, default="./dataset/data/result_np")
		# self.parser.add_argument('--model_dir', type = str, default="./result/model_line/")
		self.parser.add_argument('--model_dir', type = str, default="/log/hyp-3d-imaging/result/model_graycode")
		self.parser.add_argument('--image_formation_dir', type = str, default="./dataset/image_formation/result")
		self.parser.add_argument('--precomputed_proj_coordinates_dir', type = str, default="./dataset/image_formation/xy_vproj")
		self.parser.add_argument('--dg_intensity_dir', type = str, default='./dataset/image_formation')
		self.parser.add_argument('--dat_dir', type = str, default = './dataset/image_formation/dat')
		# self.parser.add_argument('--illum_dir', type = str, default="./dataset/image_formation/illum/line_pattern")
		self.parser.add_argument('--illum_dir', type = str, default="./dataset/image_formation/illum/graycode_pattern")
		self.parser.add_argument('--illum_data_dir', type = str, default= "./dataset/image_formation/illum_data.npy")
		self.parser.add_argument('--img_hyp_texture_dir', type = str, default="./dataset/image_formation/img_hyp_text")
		self.parser.add_argument('--random_pixel_train_dir', type = str, default="./random_datasets/random_pixel_train")
		self.parser.add_argument('--random_pixel_val_dir', type = str, default="./random_datasets/random_pixel_val")
		self.parser.add_argument('--random_pixel_eval_dir', type = str, default="./random_datasets/random_pixel_eval")
		self.parser.add_argument('--log_dir', type = str, default="./logs/")
		self.parser.add_argument('--camera_response', type = str, default="./dataset/image_formation") 
		self.parser.add_argument('--projector_response', type = str, default="./dataset/image_formation")  
		self.parser.add_argument('--random_pixel_scene_fn', type = str, default="scene_data.pt")
		self.parser.add_argument('--random_pixel_xyproj_fn', type = str, default="xy_proj_data.pt")
		self.parser.add_argument('--random_pixel_xy_real_fn', type = str, default="xy_real_data.pt")
		self.parser.add_argument('--random_pixel_hyp_norm', type = str, default="hyp_norm.pt")
		self.parser.add_argument('--random_pixel_illum', type = str, default="illum.pt")
		self.parser.add_argument('--random_pixel_hyp_gt', type = str, default="hyp_gt.pt")
		self.parser.add_argument('--real_data_dir', type=str, default="./dataset/data/real_data")

		################## TRAINING & TESTING
		self.parser.add_argument('--real_data_scene', type = bool, default= False)
		self.parser.add_argument('--feature_num', type = int, default=100)
		self.parser.add_argument('--load_dataset', action='store_true', default=False)
		self.parser.add_argument('--wvl_min', type = float, default= 420e-9) # 420e-9
		self.parser.add_argument('--wvl_max', type = float, default= 660e-9) # 660e-9
		self.parser.add_argument('--wvl_num', type = int, default= 25) # 25
		self.parser.add_argument('--noise_std', type = float, default= 0.001) 
		
		self.parser.add_argument('--m_min', type = int, default= -1) 
		self.parser.add_argument('--m_max', type = int, default= 1) 
		self.parser.add_argument('--m_num', type = int, default=3)

		self.parser.add_argument('--epoch_num', type = int, default= 2000)
		# self.parser.add_argument('--cam_W', type = int, default= 2048//2)
		# self.parser.add_argument('--cam_H', type = int, default= 1536//2)
		self.parser.add_argument('--cam_W', type = int, default= 890)
		self.parser.add_argument('--cam_H', type = int, default= 580)
		self.parser.add_argument('--proj_W', type = int, default= 1280//2) # 720
		self.parser.add_argument('--proj_H', type = int, default= 720//2)
		self.parser.add_argument('--scene_train_num', type = int, default= 200) # 200
		self.parser.add_argument('--scene_test_num', type = int, default= 20) # 20
		self.parser.add_argument('--scene_eval_num', type = int, default= 1)
		self.parser.add_argument('--scene_real_num', type = int, default= 1)

		self.parser.add_argument('--illum_num', type = int, default= 40)
		# self.parser.add_argument('--illum_num', type = int, default= 63)

		self.parser.add_argument('--patch_pixel_num', type = int, default = 9)
	
		self.parser.add_argument('--num_train_px_per_iter', type = int, default= 9*500) # 4500

		self.parser.add_argument('--batch_size_train', type = int, default= 16) # 16
		self.parser.add_argument('--batch_size_test', type = int, default= 8) # 8
		self.parser.add_argument('--batch_size_eval', type = int, default= 1)
		self.parser.add_argument('--batch_size_real', type = int, default= 1)
		self.parser.add_argument('--model_lr', type = float, default= 5*1e-4) # 5*1e-4
		# step size 300 -> 100 / 0.5 ->0.8
		self.parser.add_argument('--model_step_size', type = int, default = 300)
		self.parser.add_argument('--model_gamma', type= float, default=0.8)
  
		self.parser.add_argument('--illum_lr', type = float, default= 5*1e-4)
		self.parser.add_argument('--illum_step_size', type = int, default = 250)
		self.parser.add_argument('--illum_gamma', type= float, default=0.5)
  
		self.parser.add_argument('--weight_hyp', type = float, default= 0)
		self.parser.add_argument('--weight_depth', type=float, default= 1)
  
		################## TODO: clean the following codes

		# for init scene
		self.parser.add_argument('--N_scene', type=int, default=10)
		self.parser.add_argument('--N_obj', type=int, default=20)
		self.parser.add_argument('--bg_size', type=int, default=2.8)

		# data directory path6.6
		self.parser.add_argument('--abs')
		self.parser.add_argument('--rgb_dir', type= str, default="C:/Users/owner/Documents/GitHub/Scalable-Hyperspectral-3D-Imaging/dataset/data/rgb_texture")
  		# ================================= NEED DATA
		self.parser.add_argument('--obj_path', type=str, default="//bean.postech.ac.kr/data/ches7283/models-OBJ/models")
		self.parser.add_argument('--mat_path', type=str, default= "./dataset/m_files/")
  		# for Camera calibaration
		self.parser.add_argument('--fov', type=float, default=30.3) # 30.3d
		self.parser.add_argument('--focal_length', type=float, default= 12) # 16mm


		self.parser.add_argument('--param_path', type = str, default="./calibration/parameters")
		
  		# for Camera calibaration
		self.parser.add_argument('--sensor_width', type=float, default=5.32) # 7.18mm
		self.parser.add_argument('--sensor_height', type=float, default=5.32) # 5.32mm
		self.parser.add_argument('--baseline', type=float, default=0.1) # 10cm
		self.parser.add_argument('--cam_pitch', type=float, default=3.45*1e-6*2) # 3.45um *2 (binning)

		# projector
		self.parser.add_argument('--sensor_diag_proj',  type = float, default= 5.842) # 5.842mm
		self.parser.add_argument('--focal_length_proj', type = float, default= 8) # 8mm
		
	def parse(self):
		arg = self.parser.parse_args()
		arg.wvls = torch.linspace(arg.wvl_min, arg.wvl_max, arg.wvl_num)
		arg.m_list = torch.linspace(arg.m_min, arg.m_max, arg.m_num)
		arg.illums = torch.zeros((arg.illum_num, arg.proj_H, arg.proj_W, 3))
		return arg