use nalgebra::Matrix4;
use nalgebra::Vector4;
use nalgebra::Vector3;

use crate::view;

pub struct View {
    pub x: f64,
    pub y: f64,
    pub z: f64,
    pub roll: f64, // rotation about x axis
    pub pitch: f64, // rotation about y axis
    pub yaw: f64,  // rotation about z axis
    pub fov: f64,
    pub direction: Vector4<f64>,
    pub rotation: Matrix4<f64>,
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
        let direction: Vector4<f64> = Vector4::new(0.,0.,-1.,0.);
        let rotation: Matrix4<f64> = Matrix4::new(
            1., 0., 0., 0.,
            0., 1., 0., 0., 
            0., 0., 1., 0.,
            0., 0., 0., 1.,
        );
        View {
            x, y, z, roll, pitch, yaw, fov, direction, rotation
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
        self.rotate(val, 0., 0.,);
    }

    pub fn rotate_pitch(& mut self, val: f64){
        self.rotate(0., val, 0.,);
    }

    pub fn rotate_yaw(& mut self, val: f64){
        self.rotate(0., 0., val,);
    }

    pub fn rotate(& mut self, d_roll:f64,d_pitch: f64,d_yaw: f64){
        self.roll += d_roll;
        self.pitch += d_pitch;
        self.yaw += d_yaw;


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


        let forward_vector = Vector4::new(0.,0.,-1.,0.);

        self.rotation = x_rotation * y_rotation* z_rotation;

        self.direction = self.rotation * forward_vector;

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

        let rotated_movement_vector = self.rotation * translation;

        self.x += rotated_movement_vector[0];
        self.y += rotated_movement_vector[1];
        self.z += rotated_movement_vector[2];
    }

    pub fn print_fov(&mut self){
        
    }

    pub fn in_view(&self, point: Vector4<f64>) -> bool{
        let view_pos: Vector4<f64> = Vector4::new(self.x,self.y,self.z,1.);

        let vec_to_point = (view_pos - point).normalize();


        let forward_dir = self.direction;


        // dot prod
        let cos_a = forward_dir[0] * vec_to_point[0] + forward_dir[1] * vec_to_point[1] + forward_dir[2] * vec_to_point[2];

        

        if cos_a >= (self.fov/2.).cos() {
            return true;
        }

        return false;

    }
}