use crate::world::World;
use crate::view::View;

use crate::line::Line;

use nalgebra::Vector3;
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

        let near = 10;
        let far = 100;
        let focal_length = 10;

        let left = -10;
        let right = 10;
        let top = 10;
        let bottom = -10;


    

        // for each mesh, project their coords to a 2d plane, put lines on screen
        for mesh in &self.world.elements {

            // naive solution rn
            /*
            not gonna use homogenous coords for the time being
            also not going to cache any of the calculations
            gonna iterate over each face, I also will assume faces are triangles

             */

            for face in &mesh.faces{
                for i in 0..3 {

                    // get two IDs for vertices
                    let coord_a = face.vertices[i];
                    let coord_b = face.vertices[(i+1)%3];

                    let x1_prime: f64;
                    let y1_prime : f64;

                    let x2_prime: f64;
                    let y2_prime : f64;

                    // lots of conversions, will clean up later
                    x1_prime = (mesh.points[coord_a as usize].position[0]*near as f64) / mesh.points[coord_a as usize].position[2];
                    y1_prime = (mesh.points[coord_a as usize].position[1]*near as f64) / mesh.points[coord_a as usize].position[2];

                    x2_prime = (mesh.points[coord_b as usize].position[0]*near as f64) / mesh.points[coord_b as usize].position[2];
                    y2_prime = (mesh.points[coord_b as usize].position[1]*near as f64) / mesh.points[coord_b as usize].position[2];

                    //scalar to magnify (just for testing)

                    let line = Line::new(x1_prime*3.,y1_prime*3.,x2_prime*3.,y2_prime*3.,1.,0xFFFFFF);


                    line.draw(buffer, screen_width, screen_height);

                }
            }


        }
    }



}

