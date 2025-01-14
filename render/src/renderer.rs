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

        let near = 10.;
        let far = 100.;
        let focal_length = 10.;

        let width = screen_width as f64;
        let height = screen_height as f64;
        let depth = far-near;

        let left = -width/2.;
        let right = width/2.;
        let top = -height/2.;
        let bottom = height/2.;


    

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
            
            // i notice that there is a perspective3 matrix, idk how it works
            let c1 = (far+near) / depth;
            let c2 = 2.*far*near / depth; 
            let perspective_transformation = Matrix4::new(
                2.*near/width, 0., 0., -near*(right+left)/width,
                0., 2.*near/height, 0., -near*(top+bottom)/height,
                0., 0., c1, c2,
                0., 0., -1., 0.,
            );




            for face in &mesh.faces{
                for i in 0..3 {

                    
                    let id_a = face.vertices[i];
                    let id_b = face.vertices[(i+1)%3];

                    let point_a = &mesh.points[id_a as usize];
                    let point_b = &mesh.points[id_b as usize];

                    let mut a_position_prime = perspective_transformation * point_a.position;
                    a_position_prime /= a_position_prime[3];

                    let mut b_position_prime = perspective_transformation * point_b.position;
                    b_position_prime /= b_position_prime[3];



                    let mut x1_prime = (a_position_prime[0] +1.)/2.;
                    x1_prime *= width;

                    let mut y1_prime = (1.-a_position_prime[1])/2.;
                    y1_prime *= height;

                    let mut x2_prime = (b_position_prime[0] +1.)/2.;
                    x2_prime *= width;

                    let mut y2_prime = (1.-b_position_prime[1])/2.;
                    y2_prime *= height;


                    
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



}

