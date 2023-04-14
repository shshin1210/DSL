from generate_hyp import Hyper_Generator
from generate_rgb import RGB_Generator
import numpy as np
import os

import sys
sys.path.append('C:/Users/owner/Documents/GitHub/Scalable-Hyperspectral-3D-Imaging')
from hyper_sl.utils import ArgParser

if __name__ == "__main__":
	parser = ArgParser.Argument()
	args = parser.parse()

	N_scene = args.N_scene
	
	# bring Class 
	hyp_generator = Hyper_Generator(args)
	rgb_generator = RGB_Generator(args)

	for i in range(N_scene):
		# create random rgb texture, get rgb values
		rgb_list = hyp_generator.makeRGBtexture()
	
		# generate rgb rendered exr file
		rgb_generator.gen(i)
		
		# generate hyp numpy array
		img_hyper_text = hyp_generator.gen(i = i)
		# save img_hyper_text to numpy
		path = os.path.join(args.img_hyp_text_path, 'img_hyp_text_%04d.npy' %i)
		np.save(path,img_hyper_text)
		
		# save img to tif file : check output img
		hyp_generator.array_to_tif(img_hyper_text, i)		
	
		print(img_hyper_text.min())