use crate::point;
use crate::world::World;
use crate::view::View;

use crate::line::Line;

use nalgebra::Matrix4;
use ndarray::arr2;




/*
RIGHT HANDED COORDINATE SYSTEM:

+x = RIGHT
+y = UP
+z = FORWARD

*/

pub struct Renderer{
    pub world: World,
    pub view: View,
}

impl Renderer {
    pub fn new(world: World, view: View) -> Renderer{
        Renderer{
            world,
            view,
        }
    }

    pub fn render(&self, buffer: &mut Vec<u32>,  screen_width: i64, screen_height: i64){

        buffer.fill(0x000000);

        let near = 5.;
        let far = 100.;
        let focal_length = 10.;

        let fov = 1.222;  // ~70 degrees


        let depth = far-near;

        let left = -100.;
        let right = 100.;
        let top = -100.;
        let bottom = 100.;

        let width = right - left;
        let height = top - bottom;


    

        // for each mesh, project their coords to a 2d plane, put lines on screen
        for mesh in &self.world.elements {


            /*
            just following the matrices on this link.
            https://webgl.brown37.net/09_projections/04_projections_perspective_math.html
            */
            // used to translate the view to the center of the screen
            /*
            let translation_matrix = Matrix4::new(
                1., 0., 0., (-screen_width/2)as f64,
                0., 1., 0., (-screen_height/2)as f64,
                0., 0., 1., 0.,
                0., 0., 0., 1.,
            );

            let near_transformation = Matrix4::new(
                near, 0., 0., 0., 
                0., near, 0., 0., 
                0., 0., 1., 0., 
                0., 0., 0., 1., 
            );

            let divide_z_transformation = Matrix4::new(
                1., 0., 0., 0.,
                0.,1., 0., 0.,
                0., 0., 1., 0.,
                0., 0., -1., 0.,
            );

            let c1 = (far+near) / (near-far);
            let c2 = 2.*far*near / (near-far); 
            let scale_z_transformation = Matrix4::new(
                1., 0., 0., 0., 
                0., 1., 0., 0., 
                0., 0., c1, c2, 
                0., 0., -1., 0., 
            );
             */

            let view_translation = Matrix4::new(
                1., 0., 0., self.view.x, 
                0., 1., 0., self.view.y, 
                0., 0., 1., self.view.z, 
                0., 0., 0., 1.,
            );

            let alpha = self.view.roll;
            let x_rotation = Matrix4::new(
                1., 0., 0., 0., 
                0., alpha.cos(), -(alpha.sin()), 0.,
                0., alpha.sin(), alpha.cos(), 0., 
                0., 0., 0., 1.,
            );

            let beta = self.view.pitch;
            let y_rotation = Matrix4::new(
                beta.cos(), 0., beta.sin(), 0.,
                0., 1., 0., 0., 
                -(beta.sin()), 0., beta.cos(), 0., 
                0., 0., 0., 1.,
            );

            let gamma = self.view.yaw;
            let z_rotation = Matrix4::new(
                gamma.cos(), -(gamma.sin()), 0., 0.,
                gamma.sin(), gamma.cos(), 0., 0., 
                0., 0., 1., 0.,
                0., 0., 0., 1.,
            );


            
            // i notice that there is a perspective3 matrix, idk how it works
            let c1 = -(far+near) / depth;
            let c2 = -2.*far*near / depth; 


            // having troubles with this perspective matrix, maybe I have implemented it slightly wrong,
            // it stretches the depth quite significantly
            /*
                let perspective_transformation = Matrix4::new(
                2.*near/width, 0., 0., -near*(right+left)/width,
                0., 2.*near/height, 0., -near*(top+bottom)/height,
                0., 0., c1, c2,
                0., 0., -1., 0.,
            );
             */



            let a = screen_width as f64 / screen_height as f64;

            let tan_fov = ((fov/2.) as f64).tan();


            let perspective_transformation = Matrix4::new(
                1./(a*tan_fov), 0., 0., 0., 
                0., 1./tan_fov, 0., 0.,
                0., 0., c1, c2,
                0., 0., -1., 0.,
            );

            


           // println!("Faces");
            for face in &mesh.faces{
                for i in 0..3 {

                    
                    let id_a = face.vertices[i];
                    let id_b = face.vertices[(i+1)%3];

                    let point_a = &mesh.points[id_a as usize];
                    let point_b = &mesh.points[id_b as usize];

                    // translate based on view movement
                    let pos_a_translated = view_translation * &point_a.position;
                    let pos_b_translated = view_translation * &point_b.position;

                    // rotate x
                    let pos_a_x_rotated = x_rotation * pos_a_translated;
                    let pos_b_x_rotated = x_rotation * pos_b_translated;

                    // rotate y
                    let pos_a_y_rotated = y_rotation * pos_a_x_rotated;
                    let pos_b_y_rotated = y_rotation * pos_b_x_rotated;

                    // rotate z
                    let pos_a_z_rotated = z_rotation * pos_a_y_rotated;
                    let pos_b_z_rotated = z_rotation * pos_b_y_rotated;

                    // perspective transformation
                    let mut a_position_prime = perspective_transformation * pos_a_z_rotated;

                    let mut b_position_prime = perspective_transformation * pos_b_z_rotated;

                    if b_position_prime[3] > -near && a_position_prime[3] > -near {
                        continue;
                    }

                    a_position_prime /= a_position_prime[3];
                    b_position_prime /= b_position_prime[3];



                    // scale back to device coordinates
                    let mut x1_prime = (a_position_prime[0] +1.)/2.;
                    x1_prime *= screen_width as f64;

                    let mut y1_prime = (1.-a_position_prime[1])/2.;
                    y1_prime *= screen_height as f64;

                    let mut x2_prime = (b_position_prime[0] +1.)/2.;
                    x2_prime *= screen_width as f64;

                    let mut y2_prime = (1.-b_position_prime[1])/2.;
                    y2_prime *= screen_height as f64;


                    self.clip_at_edge(&mut x1_prime,&mut y1_prime,&mut x2_prime,&mut y2_prime,screen_width,screen_height);


                    // we need to clip the final coords to fit inside our boundary


                    //println!("{} {} {} {}", x1_prime,y1_prime,x2_prime,y2_prime);


                    let line = Line::new(x1_prime,y1_prime,x2_prime,y2_prime,1.,0xFFFFFF);




                    line.draw(buffer, screen_width, screen_height);
                    /*
                    // goodbye old way

                    // get two IDs for vertices
                    let id_a = face.vertices[i];
                    let id_b = face.vertices[(i+1)%3];

                    let mut x1_prime: f64;
                    let mut y1_prime : f64;

                    let mut x2_prime: f64;
                    let mut y2_prime : f64;

                    let point_a = &mesh.points[id_a as usize];
                    let a_x = point_a.position[0];
                    let a_y = point_a.position[1];
                    let a_z = point_a.position[2];

                    let point_b = &mesh.points[id_b as usize];
                    let b_x = point_b.position[0];
                    let b_y = point_b.position[1];
                    let b_z = point_b.position[2];

                    x1_prime = (a_x*near) / a_z;
                    y1_prime = (a_y*near) / a_z;

                    x2_prime = (b_x*near) / b_z;
                    y2_prime = (b_y*near) / b_z;

                    // translate to center of screen
                    x1_prime += (screen_width/2) as f64;
                    y1_prime += (screen_height/2) as f64;

                    x2_prime += (screen_width/2) as f64;
                    y2_prime += (screen_height/2) as f64;


                    let line = Line::new(x1_prime,y1_prime,x2_prime,y2_prime,1.,0xFFFFFF);


                    line.draw(buffer, screen_width, screen_height);

                    */

                }
            }


        }
    }

    fn clip_at_edge(&self, x1: &mut f64, y1: &mut f64, x2: &mut f64, y2: &mut f64, screen_width: i64, screen_height: i64){
        let width = screen_width as f64;
        let height =screen_height as f64;
        if *x1 < 0. {
            let scaling = *x2 / (*x2-*x1);
            let dy = *y2 - *y1;

            *x1 = 0.;
            *y1 = *y2 - (scaling * dy);
        }
        if *x2 <0. {
            let scaling = *x1 / (*x1-*x2);
            let dy = *y1 - *y2;

            *x2 = 0.;
            *y2 = *y1 - (scaling * dy);
        }

        if *x1 >= width {
            let scaling = (width - *x2) / (*x1 - *x2);
            let dy = *y2 - * y1;
            *x1 = width-1.;
            *y1 = *y2 - (scaling * dy);
        }

        if *x2 >= width {
            let scaling = (width - *x1) / (*x2 - *x1);
            let dy = *y1 - * y2;
            *x2 = width-1.;
            *y2 = *y1 - (scaling * dy);
        }


        // y
        if *y1 < 0. {
            let scaling = *y2 / (*y2-*y1);
            let dx = *x2 - *x1;

            *y1 = 0.;
            *x1 = *x2 - (scaling * dx);
        }
        if *y2 <0. {
            let scaling = *y1 / (*y1-*y2);
            let dx = *x1 - *x2;

            *y2 = 0.;
            *x2 = *x1 - (scaling * dx);
        }

        if *y1 >= height {
            let scaling = (height - *y2) / (*y1 - *y2);
            let dx = *x2 - * x1;
            *y1 = height-1.;
            *x1 = *x2 - (scaling * dx);
        }

        if *y2 >= height {
            let scaling = (height - *y1) / (*y2 - *y1);
            let dx = *x1 - * x2;
            *y2 = height-1.;
            *x2 = *x1 - (scaling * dx);
        }



    }

}

