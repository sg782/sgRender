use nalgebra::Matrix4;
use nalgebra::Vector4;

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

        let movement_vector = Vector4::new(0.,0.,val,0.);
        self.relative_move(movement_vector);        
    }
    pub fn move_side(&mut self, val: f64){

        let movement_vector = Vector4::new(val,0.,0.,0.);
        self.relative_move(movement_vector);
        
    }

    pub fn move_vertical(&mut self, val: f64){

        let movement_vector = Vector4::new(0.,val,0.,0.);
        self.relative_move(movement_vector);

    }

    pub fn relative_move(&mut self, translation: Vector4<f64>){
        let view_translation = Matrix4::new(
            1., 0., 0., -1.,
            0., 1., 0., 0.,
            0., 0., 1., 0.,
            0., 0., 0., 1.,
        );

        let alpha = -self.roll;
        let x_rotation = Matrix4::new(
            1., 0., 0., 0., 
            0., alpha.cos(), -(alpha.sin()), 0.,
            0., alpha.sin(), alpha.cos(), 0., 
            0., 0., 0., 1.,
        );

        let beta = -self.pitch;
        let y_rotation = Matrix4::new(
            beta.cos(), 0., beta.sin(), 0.,
            0., 1., 0., 0., 
            -(beta.sin()), 0., beta.cos(), 0., 
            0., 0., 0., 1.,
        );

        let gamma = -self.yaw;
        let z_rotation = Matrix4::new(
            gamma.cos(), -(gamma.sin()), 0., 0.,
            gamma.sin(), gamma.cos(), 0., 0., 
            0., 0., 1., 0.,
            0., 0., 0., 1.,
        );

        let forward_transformation = view_translation * x_rotation * y_rotation* z_rotation;

        
        let rotated_movement_vector = forward_transformation * translation;

        // println!("Vector: {}",forward_transformation * movement_vector);

        self.x += rotated_movement_vector[0];
        self.y += rotated_movement_vector[1];
        self.z += rotated_movement_vector[2];
    }

    pub fn print_fov(&mut self){
        
    }



}