use nalgebra::Matrix4;
use nalgebra::{Vector3,Vector4};
use crate::plane::Plane;
use crate::mesh::mesh::Mesh;

pub struct View {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub roll: f32, // rotation about x axis
    pub pitch: f32, // rotation about y axis
    pub yaw: f32,  // rotation about z axis
    pub fov: f32,
    pub near: f32,
    pub far: f32,
    pub aspect_ratio: f32, 
    pub direction: Vector4<f32>,
    pub rotation: Matrix4<f32>,
    pub frustum_faces: Vec<Plane>,
}

impl View{
    pub fn new(
        x: f32,
        y: f32,
        z: f32,
        roll: f32, // rotation about x axis
        pitch: f32, // rotation about y axis
        yaw: f32,  // rotation about z axis
        fov: f32,
        screen_width: usize,
        screen_height: usize,
    ) -> View {
        let direction: Vector4<f32> = Vector4::new(0.,0.,-1.,0.);
        let rotation: Matrix4<f32> = Matrix4::new(
            1., 0., 0., 0.,
            0., 1., 0., 0., 
            0., 0., 1., 0.,
            0., 0., 0., 1.,
        );

        // hardcode for now cuz i dont wanna refactor previous instantations haha will do later
        let near = 0.1;
        let far = 500.;

        // ratio of width to height
        let aspect_ratio = screen_width as f32 / screen_height as f32;

        let frustum_faces: Vec<Plane> = Vec::new();

        let mut view = View {
            x, y, z, roll, pitch, yaw, fov, near, far, aspect_ratio, direction, rotation, frustum_faces
        };

        view.calculate_frustum_planes();

        view
    }

    // pub fn move_x(& mut self, val: f32){
    //     self.x += val;
    // }

    // pub fn move_y(& mut self, val: f32){
    //     self.y += val;
    // }

    // pub fn move_z(& mut self, val: f32){
    //     self.z += val;
    // }

    pub fn rotate_roll(& mut self, val: f32){
        self.rotate(val, 0., 0.,);
    }

    pub fn rotate_pitch(& mut self, val: f32){
        self.rotate(0., val, 0.,);
    }

    pub fn rotate_yaw(& mut self, val: f32){
        self.rotate(0., 0., val,);
    }

    pub fn rotate(& mut self, d_roll:f32,d_pitch: f32,d_yaw: f32){
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

        self.calculate_frustum_planes();


    }

    pub fn move_forward(&mut self, val: f32){

        let movement_vector = Vector4::new(0.,0.,val,0.);
        self.relative_move(movement_vector);        
    }
    pub fn move_side(&mut self, val: f32){

        let movement_vector = Vector4::new(val,0.,0.,0.);
        self.relative_move(movement_vector);
        
    }

    pub fn move_vertical(&mut self, val: f32){

        let movement_vector = Vector4::new(0.,val,0.,0.);
        self.relative_move(movement_vector);

    }

    pub fn relative_move(&mut self, translation: Vector4<f32>){

        let rotated_movement_vector = self.rotation * translation;

        

        self.x += rotated_movement_vector[0];
        self.y += rotated_movement_vector[1];
        self.z += rotated_movement_vector[2];

        self.calculate_frustum_planes();
    }


    pub fn calculate_translated_position_relative(&mut self, translation: Vector4<f32>) -> Vector3<f32> {
        let rotated_movement_vector = self.rotation * translation;

        let out:Vector3<f32> = Vector3::new(
            self.x + rotated_movement_vector[0],
            self.y+rotated_movement_vector[1],
            self.z+rotated_movement_vector[2],
        );

        return out;
    }

    // pub fn print_fov(&mut self){
        
    // }

    pub fn in_view(&self, mesh: &Box<dyn Mesh>) -> bool{

        let bounding_box = mesh.bounding_box();


        // if any are in every plane, then good


        for plane in &self.frustum_faces {
            //println!("Idx; {}", idx);
            let mut all_points_outside = true;


            // iterate nicely over
            

            for i in 0..8{
                let cur_point: Vector3<f32> = Vector3::new(
                    if i & 1 == 0 {bounding_box[0][0]} else {bounding_box[1][0]},
                    if i & 2 == 0 {bounding_box[0][1]} else {bounding_box[1][1]},
                    if i & 4 == 0 {bounding_box[0][2]} else {bounding_box[1][2]},
                );


                if plane.is_inside(cur_point){
                   all_points_outside = false;
                    break;
                }
            }

            if all_points_outside {
                //println!("Culled by {}", idx);
                return false;
            }

        }

        true

    }

    pub fn calculate_frustum_planes (&mut self){

        // leaving in the 4th vector point to avoid having to redefine direciton as well
        let position: Vector3<f32> = Vector3::new(self.x,self.y,self.z);

        let heading: Vector3<f32> = Vector3::new(self.direction[0],self.direction[1],self.direction[2]);

        let near_center = position + (heading * self.near);
        let far_center = position + (heading * self.far);



        
        //can be optimized, many duplicate calculations. once i get it working, I will do thatt

        // define near vertices
        // tl = top left, br = bottom right and so on

        let mut near_tl = near_center.clone();
        near_tl[1] -= (self.fov/2.).tan() * self.near;
        near_tl[0] -= self.aspect_ratio * (self.fov / 2.).tan() * self.near;

        let mut near_tr = near_center.clone(); //here
        near_tr[1] -= (self.fov/2.).tan() * self.near;
        near_tr[0] += self.aspect_ratio * (self.fov / 2.).tan() * self.near;

        let mut near_bl = near_center.clone();
        near_bl[1] += (self.fov/2.).tan() * self.near;
        near_bl[0] -= self.aspect_ratio * (self.fov / 2.).tan() * self.near;

        let mut near_br = near_center.clone(); // here
        near_br[1] += (self.fov/2.).tan() * self.near;
        near_br[0] += self.aspect_ratio * (self.fov / 2.).tan() * self.near;

        

        // define far vertices
        let mut far_tl = far_center.clone();
        far_tl[1] -= (self.fov/2.).tan() * self.far;
        far_tl[0] -= self.aspect_ratio * (self.fov / 2.).tan() * self.far;

        let mut far_tr = far_center.clone(); // here
        far_tr[1] -= (self.fov/2.).tan() * self.far;
        far_tr[0] += self.aspect_ratio * (self.fov / 2.).tan() * self.far;

        let mut far_bl = far_center.clone();
        far_bl[1] += (self.fov/2.).tan() * self.far;
        far_bl[0] -= self.aspect_ratio * (self.fov / 2.).tan() * self.far;

        let mut far_br = far_center.clone();
        far_br[1] += (self.fov/2.).tan() * self.far;
        far_br[0] += self.aspect_ratio * (self.fov / 2.).tan() * self.far;



        self.frustum_faces.clear();

        let fov = self.fov/2.;


        // i forgot that rotations always add some complexity. I did not account for them, new idea
        let half_near_height = (fov).tan() * self.near;
        let half_near_width = self.aspect_ratio * half_near_height;

        let half_far_height =  (fov).tan() * self.far;
        let half_far_width = self.aspect_ratio * half_far_height;

        // now we MOVE the camera to the right position rq, and get a ray from there
        // we will test with the right plane only, so we need near_tr, near_br, far_tr
        // this is prolly slow rn, since im using other functions, but it can be optmized later



        // near_tr
        let translation: Vector4<f32> = Vector4::new(half_near_width,-half_near_height,-self.near,0.);
        let near_tr: Vector3<f32> = self.calculate_translated_position_relative(translation);

        // near_br
        let translation: Vector4<f32> = Vector4::new(half_near_width,half_near_height,-self.near,0.);
        let near_br: Vector3<f32> = self.calculate_translated_position_relative(translation);

        // near_tl
        let translation: Vector4<f32> = Vector4::new(-half_near_width,-half_near_height,-self.near,0.);
        let near_tl: Vector3<f32> = self.calculate_translated_position_relative(translation);

        // near_bl
        let translation: Vector4<f32> = Vector4::new(-half_near_width,half_near_height,-self.near,0.);
        let near_bl: Vector3<f32> = self.calculate_translated_position_relative(translation);


        // far_tr
        let translation: Vector4<f32> = Vector4::new(half_far_width,-half_far_height,-self.far,0.);
        let far_tr: Vector3<f32> = self.calculate_translated_position_relative(translation);

        // far_br
        let translation: Vector4<f32> = Vector4::new(half_far_width,half_far_height,-self.far,0.);
        let far_br: Vector3<f32> = self.calculate_translated_position_relative(translation);

        // far_tl
        let translation: Vector4<f32> = Vector4::new(-half_far_width,-half_far_height,-self.far,0.);
        let far_tl: Vector3<f32> = self.calculate_translated_position_relative(translation);

        // far_bl
        let translation: Vector4<f32> = Vector4::new(-half_far_width,half_far_height,-self.far,0.);
        let far_bl: Vector3<f32> = self.calculate_translated_position_relative(translation);




        // fix faces for use in the right hand system

        //near
        self.frustum_faces.push(Plane::from_points(near_bl, far_bl, near_br)); 

        //far
        self.frustum_faces.push(Plane::from_points(far_tr,  far_br, far_tl)); 

        // left
        self.frustum_faces.push(Plane::from_points(far_tl,far_bl, near_tl)); // good

        // right
        self.frustum_faces.push(Plane::from_points(near_tr, near_br, far_tr)); // good


        //top  (somewhere I must have flipped up and down, i assume the bug is in line.draw, where i forget to flip the 2d y-axis. 
        // top and bottom work fine, they just are flipped
        self.frustum_faces.push(Plane::from_points(near_tr, far_tr, near_tl)); //good

        // bottom
        self.frustum_faces.push(Plane::from_points(near_br, near_bl, far_br)); //good




    }

}