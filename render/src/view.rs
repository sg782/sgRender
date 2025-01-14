pub struct View {
    pub x: f64,
    pub y: f64,
    pub z: f64,
    pub roll: f64, // rotation about x axis
    pub pitch: f64, // rotation about y axis
    pub yaw: f64,  // rotation about z axis
    pub fov: f64,
}

impl View{
    pub fn new(
        x: f64,
        y: f64,
        z: f64,
        roll: f64, // rotation about x axis
        pitch: f64, // rotation about y axis
        yaw: f64,  // rotation about z axis
        fov: f64,
    ) -> View {
        View {
            x, y, z, roll, pitch, yaw, fov
        }
    }

    pub fn move_x(& mut self, val: f64){
        self.x += val;
    }

    pub fn move_y(& mut self, val: f64){
        self.y += val;
    }

    pub fn move_z(& mut self, val: f64){
        self.z += val;
    }

    pub fn rotate_roll(& mut self, val: f64){
        self.roll += val;
    }

    pub fn rotate_pitch(& mut self, val: f64){
        self.pitch += val;
    }

    pub fn rotate_yaw(& mut self, val: f64){
        self.yaw += val;
    }

    pub fn move_forward(&mut self, val: f64){
        
    }
    pub fn move_side(&mut self, val: f64){
        
    }

    pub fn print_fov(&mut self){
        
    }



}