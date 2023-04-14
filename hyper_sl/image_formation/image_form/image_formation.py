from pattern_form import Pattern_formation
from hyptext_form import HyperText_formation

import sys
sys.path.append('C:/Users/shins/HyperImaging/Dataset')
from utils import ArgParser

if __name__ == "__main__":
    parser = ArgParser.Argument()
    args = parser.parse()

    N_scene = 1

    # bring Class 
    patt_formation = Pattern_formation(args)
    hyp_formation = HyperText_formation(args)

    for i in range(N_scene):
        # pattern formation (pattern & occlusion map)
        illum_w_occ = patt_formation.patt_form(i)

        # # save oc 
        # illum_w_occ_np = illum_w_occ.numpy()
        # cv2.imwrite(f'./illum_w_occ_{i}.png', illum_w_occ_np)
        
        # hypertexture formation
        img_hyp_txt = hyp_formation.hyp_form(illum_w_occ, i)

        # save as tif file to check result
        hyp_formation.array_to_tif(img_hyp_txt, i)

        # dispersion formation
        print('end')