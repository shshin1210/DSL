import bpy
import random, math,os
import numpy as np
from camera import Camera


"""
Need to create many scenes
make background
place camera
"""

class RGB_Generator:
    def __init__(self, arg):
        self.N_scene = arg.N_scene
        self.N_obj = arg.N_obj

        self.obj_path = arg.obj_path
        self.output_dir = arg.output_dir
        self.rgb_dir = arg.rgb_dir

        self.pi = np.pi / 180.0

        self.bg_size = arg.bg_size
        self.baseline = arg.baseline
        self.focal_length = arg.focal_length *1e-3
        self.sensor_height = arg.sensor_height *1e-3
        self.sensor_width = arg.sensor_width*1e-3
        self.cam_W, self.cam_H = arg.cam_W, arg.cam_H
        self.fov = arg.fov
    
    def gen(self, i):
        self.scene_init()

        self.clean_up_scene()

        size = self.create_background()
        self.gen_RGB_obj()
        
        # Albedo
        self.render("Albedo",i)
        self.links.clear()

        # # Depth map
        # self.render("Depth",i)
        # self.links.clear()

        # # Normal
        # self.render("Normal", i)
        # self.links.clear()

        # # Occlusion
        # self.render("Occlusion", i)
        # self.links.clear()


    def scene_init(self):
        """ Create Scene

        place camera & light        
        """
        self.scene = bpy.context.scene
        
        # # to reduce noise?
        # self.scene.cycles.samples = 1000
        # self.scene.cycles.sample_clamp_indirect = 1.0

        # nodes for rendering
        self.scene.use_nodes = True
        self.nodes = self.scene.node_tree.nodes
        self.links = self.scene.node_tree.links

        # add camera
        self.size = self.bg_size
        
        # Camera face at 0,0,0, rotation : 0,0,0
        Camera0 = Camera(self.fov ,self.focal_length, (0,0,0), (0, 0, 0), "camera0", self.sensor_width, self.sensor_height)
        self.camList = [Camera0]

        for cam in self.camList:
            self.scene.collection.objects.link(cam.get_cam_obj())
        
        # light / projector
        light = bpy.data.objects['Light']
        light.location = (0.0525938738,-0.01156695638,-0.00697808189)
        # light.location = (3.3431,-0.011567,0.006978)
        light.data.shadow_soft_size = 0
        light.rotation_euler = (self.pi*37.3,self.pi*3.16,self.pi*107)
        # light.data.cycles.cast_shadow = False

    def clean_up_scene(self):
        """ Clean up the scene

            remove defualts        
        """
        for node in self.nodes:
            self.nodes.remove(node)

        for mat in bpy.data.materials:
            mat.user_clear()
            bpy.data.materials.remove(mat)
        
        for texture in bpy.data.textures:
            texture.uesr_clear()
        
        bpy.ops.object.select_all(action = "DESELECT")

        for item in bpy.data.objects:
            if item.type == 'MESH':
                bpy.data.objects[item.name].select_set(True)
                bpy.ops.object.delete()

        for item in bpy.data.meshes:
            bpy.data.meshes.remove(item)


    def create_background(self):
        """     Create background (cube)

        Return : size / max size of background        
        
        """
        self.size = self.bg_size

        bpy.ops.mesh.primitive_plane_add(enter_editmode=False, align='WORLD', location=(0,0,-self.size/2), size = self.size, rotation = (0,0,0))

        text_list = os.listdir(self.rgb_dir)
        text_choice = text_list[0]

        plane_name = []

        for obj_name in bpy.data.objects.keys():
            if "Plane" not in obj_name:
                continue
            else:
                plane_name.append(obj_name)

        # for obj_name, text_name in zip(plane_name, text_choice):
        #     obj = bpy.data.objects[obj_name]
        #     text_path = os.path.join(self.rgb_dir, text_name)
        #     self.apply_texture_to_object(text_path, obj)

        for obj_name in plane_name:
            obj = bpy.data.objects[obj_name]
            text_path = os.path.join(self.rgb_dir, text_choice)
            self.apply_texture_to_object(text_path, obj)

        return self.size
    
    # def apply_emission_to_bg(self, tex_path, obj):
    #             # Create material for texture
    #     mat = bpy.data.materials.new(obj.data.name + '_texture')
    #     obj.data.materials.clear()
    #     obj.data.materials.append(mat)

    #     mat.use_nodes = True 
    #     mat_nodes = mat.node_tree.nodes 

    #     bsdf_node = mat_nodes['Principled BSDF']
    #     mat_out_node = mat_nodes['Material Output']

    #     tex_node = mat_nodes.new('ShaderNodeTexImage')
    #     tex_img = bpy.data.images.load(tex_path)
    #     tex_node.image = tex_img 

    #     # Material shading node linking
    #     mat.node_tree.links.new(tex_node.outputs['Color'], bsdf_node.inputs['Base Color'])
    #     mat.node_tree.links.new(bsdf_node.outputs['BSDF'], mat_out_node.inputs['Surface'])
    #     # bsdf_node.inputs[1].default_value = 0

    #     for i in range(len(list(bsdf_node.inputs))):
    #         print(bsdf_node.inputs[i].name, i)

    #     # bsdf_node.inputs[19].default_value = 0
    #     # bpy.data.materials["Material"].node_tree.nodes["Emission"].inputs[1].default_value = 1


    def apply_texture_to_object(self, tex_path, obj):
        """     Randomly import image texture and add to the object
        Args:
            tex_path : directory path containing texture img files
            obj : a target bpy object to use the texture
            
        """
                
        # Create material for texture
        mat = bpy.data.materials.new(obj.data.name + '_texture')
        obj.data.materials.clear()
        obj.data.materials.append(mat)

        mat.use_nodes = True 
        mat_nodes = mat.node_tree.nodes 

        bsdf_node = mat_nodes['Principled BSDF']
        mat_out_node = mat_nodes['Material Output']

        tex_node = mat_nodes.new('ShaderNodeTexImage')
        tex_img = bpy.data.images.load(tex_path)
        tex_node.image = tex_img 

        # Material shading node linking
        mat.node_tree.links.new(tex_node.outputs['Color'], bsdf_node.inputs['Base Color'])
        mat.node_tree.links.new(bsdf_node.outputs['BSDF'], mat_out_node.inputs['Surface'])

    def gen_RGB_obj(self):
        """     Generate random objects from shapeNet
        
        randomly import obj files and put/rotate/resize them on scene
        and randomly paint with randomRGB pairs

        return : return rendered RGB png
        """

        # generate ShapeNet objects
        obj_list = os.listdir(self.obj_path)
        obj_list = [file for file in obj_list if file.endswith(".obj")] # obj file list
        obj_choice = random.sample(obj_list, self.N_obj) # randomly pick obj files

        text_list = os.listdir(self.rgb_dir)
        text_choice = text_list[1:self.N_obj+1]

        radian_360 = math.radians(360)

        """
        generate random objects
        """
        # random scale & ratation for each objs
        for (obj_name, tex_name) in zip(obj_choice, text_choice):
            prev_objects = set(self.scene.objects.keys())

            path = os.path.join(self.obj_path, obj_name)
            bpy.ops.import_scene.obj(filepath = path)
            imported_obj_name = list(set(self.scene.objects.keys()) - prev_objects)[0]
            imported_obj = bpy.data.objects[imported_obj_name] 
            
            # ============================ NEED TO CHANGE LOCATION ==========================================

            # obj location
            loc_x, loc_y, loc_z = random.uniform(-self.size/10,-self.size/10), random.uniform(0, self.size/10), random.uniform(-self.size/10, self.size/10)
            imported_obj.location = (random.choice([1,-1])*loc_x, random.choice([0,1])* loc_y, random.choice([1,-1])* loc_z)

            x, y, z= random.uniform(-self.size/12, self.size/12), random.uniform(-self.size/12, self.size/12), random.uniform(-1,-1.35)

            imported_obj.location = (x, y, z)
            
            # scale_x, scale_y, scale_z = random.uniform(0.1, 0.5)*z, random.uniform(0.1, 0.5)*z, random.uniform(0.1, 0.5)*z
            scale_x, scale_y, scale_z = random.uniform(0.001, 0.005), random.uniform(0.001, 0.005), random.uniform(0.001, 0.005)
            
            # rotation & scale
            imported_obj.scale = (scale_x, scale_y, scale_z)
            imported_obj.rotation_euler = (random.uniform(0, radian_360), random.uniform(0, radian_360), random.uniform(0, radian_360))

            imported_tex_path = os.path.join(self.rgb_dir, tex_name)
            self.apply_texture_to_object(imported_tex_path, imported_obj)

    def render_init(self, render_type):
        """ Rendering Initialization

        engine : Cycles
        
        """

        # if render_type == "Occlusion":
        #     self.scene.render.image_settings.file_format = "PNG"
        
        # else:
        #     self.scene.render.image_settings.file_format = "OPEN_EXR"

        self.scene.render.image_settings.file_format = "OPEN_EXR"
        self.scene.render.engine = "CYCLES"

        # rendering image resolution
        self.render_layer_node = self.nodes.new('CompositorNodeRLayers')
        self.compositor_node = self.nodes.new('CompositorNodeComposite')
        self.map_value_node = self.nodes.new('CompositorNodeMapValue')
        self.normalize_value_node = self.nodes.new('CompositorNodeNormalize')

        # normal map node
        self.seperate_RGBA_node = self.nodes.new("CompositorNodeSepRGBA")

        self.add_node_R = self.nodes.new("CompositorNodeMath")
        self.add_node_R.operation = "ADD"
        self.add_node_R.inputs[1].default_value = 1

        self.add_node_G = self.nodes.new("CompositorNodeMath")
        self.add_node_G.operation = "ADD"
        self.add_node_G.inputs[1].default_value = 1

        self.add_node_B = self.nodes.new("CompositorNodeMath")
        self.add_node_B.operation = "ADD"
        self.add_node_B.inputs[1].default_value = 1

        self.divide_node_R = self.nodes.new("CompositorNodeMath")
        self.divide_node_R.operation = "DIVIDE"
        self.divide_node_R.inputs[1].default_value = 2

        self.divide_node_G = self.nodes.new("CompositorNodeMath")
        self.divide_node_G.operation = "DIVIDE"
        self.divide_node_G.inputs[1].default_value = 2

        self.divide_node_B = self.nodes.new("CompositorNodeMath")
        self.divide_node_B.operation = "DIVIDE"
        self.divide_node_B.inputs[1].default_value = 2

        self.combine_RBGA_node = self.nodes.new("CompositorNodeCombRGBA")

        self.alpha_node = self.nodes.new("CompositorNodeSetAlpha")

        # Depth node
        self.map_value_node.offset[0] = 0
        self.map_value_node.size[0] = 0.1
        self.map_value_node.use_min = True
        self.map_value_node.use_max = True
        self.map_value_node.min[0] = 0.0
        self.map_value_node.max[0] = 255.

    def render(self, render_type, i):
        """     Rendering type
                render type : albedo, depth, occlusion, normal
        """
        self.render_init(render_type)

        if render_type == "Albedo":
        # Albedo render type
            bpy.context.view_layer.use_pass_diffuse_color = True
            self.links.new(self.render_layer_node.outputs['DiffCol'], self.alpha_node.inputs["Image"])
            self.links.new(self.render_layer_node.outputs["Alpha"], self.alpha_node.inputs['Alpha'])
            self.links.new(self.alpha_node.outputs["Image"], self.compositor_node.inputs[0])

        elif render_type == "Depth":
        # Depth render type
            bpy.context.view_layer.use_pass_z = True 
            # self.links.new(self.render_layer_node.outputs['Depth'], self.normalize_value_node.inputs[0])
            # self.links.new(self.normalize_value_node.outputs[0], self.compositor_node.inputs[0])
            self.links.new(self.render_layer_node.outputs['Depth'], self.map_value_node.inputs[0])
            self.links.new(self.map_value_node.outputs[0], self.compositor_node.inputs[0])

        elif render_type == "Normal":
            bpy.context.view_layer.use_pass_normal = True
            # R
            self.links.new(self.render_layer_node.outputs["Normal"], self.seperate_RGBA_node.inputs[0])
            self.links.new(self.seperate_RGBA_node.outputs['R'], self.add_node_R.inputs["Value"])
            self.links.new(self.add_node_R.outputs["Value"], self.divide_node_R.inputs['Value'])
            self.links.new(self.divide_node_R.outputs["Value"], self.combine_RBGA_node.inputs['R'])
            self.links.new(self.combine_RBGA_node.outputs[0], self.compositor_node.inputs[0])

            # G
            self.links.new(self.seperate_RGBA_node.outputs['G'], self.add_node_G.inputs["Value"])
            self.links.new(self.add_node_G.outputs["Value"], self.divide_node_G.inputs['Value'])
            self.links.new(self.divide_node_G.outputs["Value"], self.combine_RBGA_node.inputs['G'])
            self.links.new(self.combine_RBGA_node.outputs[0], self.compositor_node.inputs[0])

            # B
            self.links.new(self.seperate_RGBA_node.outputs['B'], self.add_node_B.inputs["Value"])
            self.links.new(self.add_node_B.outputs["Value"], self.divide_node_B.inputs['Value'])
            self.links.new(self.divide_node_B.outputs["Value"], self.combine_RBGA_node.inputs['B'])
            self.links.new(self.combine_RBGA_node.outputs[0], self.compositor_node.inputs[0])
        
        # Occlusion
        else: 
            bpy.context.view_layer.use_pass_shadow = True
            self.links.new(self.render_layer_node.outputs['Shadow'], self.compositor_node.inputs[0])
            # self.links.new(self.math_node.outputs[0],self.compositor_node.inputs[0])
            # self.math_node.inputs[1].default_value = 0.4

        # make 512x512
        self.scene.render.resolution_x = self.cam_W
        self.scene.render.resolution_y = self.cam_H

        img_name = "scene_%04d_%s"%(i, render_type)

        for cam in self.camList:
            # img_path = os.path.join(self.output_dir, os.path.join(cam.get_cam_name(), img_name))
            img_path = os.path.join(self.output_dir, img_name)
            self.scene.render.filepath = img_path
            self.scene.camera = cam.get_cam_obj()
            bpy.ops.render.render(write_still = True)
            
