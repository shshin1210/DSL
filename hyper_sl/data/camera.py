import bpy

# Camera((x,y,z), (x,y,z), "cam0")
class Camera:
    def __init__(self,fov, focal_length, location, rotation, name, sensor_width, sensor_height):
    # def __init__(self, location, rotation, name):
        self.location = location
        self.rotation = rotation
        self.name = name

        self.sensor_width = sensor_width
        self.sensor_height = sensor_height
        self.fov = fov

        self.focal_length = focal_length
        # self.fov = fov # camera.angle

        self.create_camera()

    def create_camera(self):
        # creating new camera
        camera_data = bpy.data.cameras.new(name = self.name)
        camera_obj = bpy.data.objects.new(self.name, camera_data)
        self.obj = camera_obj

        self.obj.location = self.location
        self.obj.rotation_euler = self.rotation

        self.obj.data.lens_unit = "MILLIMETERS"

        self.obj.data.angle = self.focal_length

        self.obj.data.sensor_width = self.sensor_width
        self.obj.data.sensor_height = self.sensor_height

        self.set_lens_type()
        
    def set_lens_type(self):
        self.obj.data.lens = self.focal_length

    def get_cam_obj(self):
        return self.obj

    def get_cam_name(self):
        return self.name
