import numpy as np
import random, scipy.io, cv2
import OpenEXR, Imath, os
import bpy
import tifffile


class Hyper_Generator:       

    def __init__(self, arg):

        self.N_obj = arg.N_obj
        self.N_scene = arg.N_scene
        self.wvls_n = arg.wvl_num
        
        self.mat_path = arg.mat_path
        self.output_dir = arg.output_dir
        self.rgb_dir = arg.rgb_dir

        self.scene = bpy.context.scene   

    def gen(self, i):
        
        cube_dict = self.bringHyperTexture()
        img = self.read_exr_as_np(i, "Albedo")
        img_hyper_text = self.apply_Hyper_texture(cube_dict, img)
        
        return img_hyper_text

    def makeRGBtexture(self):
        """     Create random RGB colors

        return : random RGB array
        """
        try:    
            text_list = os.listdir(self.rgb_dir)

            for i in range(len(text_list)):
                file_path = os.path.join(self.rgb_dir, text_list[i])
                os.remove(file_path)
        except:
            pass    
            
        H,W = 512,512
        rgb_list = np.zeros(shape = (self.N_obj+1,3))
        rgb_texture = np.zeros(shape = (H,W,3))

        for i in range(self.N_obj+1):
            for j in range(3):
                rgb_value = (random.randint(0,255))
                rgb_texture[:,:,j] = rgb_value
                # rgb_list[i][2-j] = rgb_value/255
                rgb_list[i][j] = rgb_value/255
            cv2.imwrite(self.rgb_dir + f"/rgb_{i}.jpg", rgb_texture)
        
        return rgb_list

    def bringHyperTexture(self):

        """     Bring (N_obj + 6(plane faces)) Hyperspectral texture randomly in dict type 

        Return : hypertexture dictionary {'texture0':np.array, 'texture1':np.array, ... } 
        """

        rand_list = []
        mat_list = []
        cube_dict = {}

        # create random number 1~60
        for i in range(self.N_obj+2):
            rand_num = random.randint(0,59)
            rand_list.append(rand_num)

        for i in range(len(rand_list)):
            if rand_list[i] < 10:
                mat_file = self.mat_path + f"T0{rand_list[i]}array.mat"
            else:
                mat_file = self.mat_path + f"T{rand_list[i]}array.mat"
            # mat_list.append(mat_file[:,:,:29])
            mat_list.append(mat_file)
        
        for i in range(len(mat_list)):
            mat = scipy.io.loadmat(mat_list[i])
            name = f"texture{i}"
            cube_dict[name] = mat['cube'][:,:,:29]

        return cube_dict


    def read_exr_as_np(self, i, render_type):
        """  Read exr file(rgb rendered obj scene) as numpy array
        
        exr file shape RGBA 640x640x4
        return : numpy array        
        """
        
        fn = os.listdir(self.output_dir)
        fn = [file for file in fn if render_type in file]
        
        fn = os.path.join(self.output_dir, fn[i])

        f = OpenEXR.InputFile(fn)
        channels = f.header()['channels']
 
        dw = f.header()['dataWindow']
        size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)

        ch_names = []

        image = np.zeros((size[1], size[0], len(channels)-1))

        for i, ch_name in enumerate(channels):
            ch_names.append(ch_name)
            ch_dtype = channels[ch_name].type
            ch_str = f.channel(ch_name, ch_dtype)

            if ch_dtype == Imath.PixelType(Imath.PixelType.FLOAT):
                np_dtype = np.float32
            elif ch_dtype == Imath.PixelType(Imath.PixelType.HALF):
                np_dtype = np.half

            image_ch = np.fromstring(ch_str, dtype=np_dtype)
            image_ch.shape = (size[1], size[0])

            if ch_name == "A" :
                continue
            else:
                image[:,:,3-i] = image_ch
        
        return image

    def apply_Hyper_texture(self, cube_dict, img):
        """     Apply hyper texture to RGB rendered img (from gen_RGB_obj)

        return np.array of img
        """
        H,W = img.shape[:2]
        a = np.zeros(shape = (H,W, self.wvls_n))

        # Get Masked RGB list
        rgb_list = np.zeros(shape=(self.N_obj+1,3))

        img = np.round(img, 5) # to 4 decimals
        img_reshape = img

        img_reshape = img_reshape.reshape(H*W,3)
        unique, cnts = np.unique(img_reshape, return_counts=True, axis=0)

        cnts_sorted = np.sort(cnts)[::-1]

        k = 0
        for i in range(len(cnts)):
            if cnts[i] > cnts_sorted[self.N_obj +1]:
                rgb_list[k] = unique[i]
                k += 1

        # # Hyperspectral Texture to masked id's
        # for k in range(self.N_obj+1):
        #     for i in range(H):
        #         for j in range(W):
        #             if np.allclose(img[i][j], rgb_list[k], rtol=1e-3, atol=1e-8):
        #                 a[i][j] = cube_dict[f'texture{k}'][i][j]
        
        # # fill up 0
        # for i in range(H):
        #     for j in range(W):
        #         if (a[i][j] == 0.0).all():
        #             a[i][j] =  cube_dict[f'texture{self.N_obj+1}'][i][j]


        # masking
        a_reshape = a.reshape(H*W, self.wvls_n)

        for i in range(self.N_obj+1):
            mask = np.isclose(img_reshape[:,:], rgb_list[i,:], rtol = 1e-3, atol = 1e-8)
            # mask = img_reshape[:,:] == rgb_list[i,:]
            
            # make mask[row,:] all True
            for k in range(mask.shape[0]):
                if not np.all(mask[k,:]): 
                    mask[k,:] = [False,False,False]

            # mask reshape to 640*640,77
            for k in range(9):
                if k == 0:
                    mask_ap = np.append(mask, mask, axis =1)
                else:
                    mask_ap = np.append(mask_ap, mask, axis = 1)
            mask_ap = np.delete(mask_ap, -1, axis = 1)

            a_reshape = np.ma.array(a_reshape, mask = mask_ap)

            # fill
            hyp_text_reshape = cube_dict[f'texture{i}'].reshape(640*640,self.wvls_n)
            a_reshape = a_reshape.filled(fill_value = hyp_text_reshape[:,:])
        
        # fill up 0
        hyp_text_reshape_bg = cube_dict[f'texture{self.N_obj+1}'].reshape(640*640, self.wvls_n)

        for i in range(a_reshape.shape[0]):
            if (a_reshape[i,:] == 0.0).all():
                a_reshape[i,:] = hyp_text_reshape_bg[i,:]

        a_reshape = a_reshape.reshape(H,W, self.wvls_n)

        img_hyper_text = a_reshape

        return img_hyper_text

    def array_to_tif(self, arr, i):
        """     Make array into tif file
        
        check hyperspectral image / tif file
        """
        arr = np.swapaxes(arr,0,2)
        arr = np.swapaxes(arr,1,2)
        # tifffile.imwrite(f'./tif_file/output_scene_{i}.tif', arr)
        file_path = os.path.join(self.output_dir, 'scene_%04d_hyp_text.tif' %(i))
        tifffile.imwrite(file_path, arr)