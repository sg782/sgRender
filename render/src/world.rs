
use nalgebra::Vector3;

use crate::lighting::directional_light::{self, DirectionalLight};
use crate::lighting::point_light::{self, PointLight};
use crate::mesh::mesh::Mesh;
use crate::mesh::point;
use crate::models::cube::Cube;
use crate::models::rect_prism::RectPrism;

use crate::models::graphical_plane::GraphicalPlane;

use crate::models::imported::Imported;

pub struct Lights {
    pub directional_lights: Vec<DirectionalLight>,
    pub point_lights: Vec<PointLight>,
}

pub struct World{
    pub elements: Vec<Box<dyn Mesh>>,
    pub lights: Lights,
    pub idx_vec_running: Vec<i32>,
}

impl World{

    pub fn new() -> World {


        /*
        ADJUST 'amount_high' and 'amount_wide' for quick changes in the models.
        you can also make a custom shape of any kind if you implement it with 'mesh' traits
         */

        let mut directional_lights = Vec::new();
        let mut point_lights = Vec::new();

        point_lights.push(PointLight::new(Vector3::new(0.,0.,1000.), 0.3));

        point_lights.push(PointLight::new(Vector3::new(0.,-20.,0.), 0.1));


        let lights = Lights {
            directional_lights,
            point_lights,
        };

        let mut elements: Vec<Box<dyn Mesh>> = Vec::new();

        // //test with a row of cubes
        let side_length = 5.;
        let amount_wide = 2;
        let amount_high = 2;
        let amount_deep = 2;
        let spacing = 10.;
        let mut count: f32 = 0.;
        for i in 0..amount_wide{
            for j in 0..amount_high{
                for k in 0..amount_deep{
                    count +=1.;
                    let idx = i as f32;
                    let jdx = j as f32;
                    let kdx = k as f32;
                    let r = 4.* (4.1+kdx);
                    //let cube = Cube::new(r * idx,jdx * jdx, r * kdx, side_length, 0xA028 * count as u32);
                    let cube = Cube::new(idx * (side_length + spacing),jdx * (side_length + spacing), kdx * (side_length + spacing), side_length-1., 0xC028 * count as u32);

                    // let teapot = Imported::new("../../3d_models/teapot.obj",side_length,idx * side_length, jdx * side_length, kdx*side_length);

                    // elements.push(Box::new(teapot));

                    elements.push(Box::new(cube));

                }

            }
        }


        // let prism = RectPrism::new(0.,0.,0.,10.,2.,2.5, 0x00FF00);
        // elements.push(Box::new(prism));

        //println!("NUM items; {}", count);

        // let v0: Vector3<f32> = Vector3::new(10.,20.,14.);
        // let v1: Vector3<f32> = Vector3::new(30.,50.,6.);

        // // // // let v2: Vector3<f32> = Vector3::new(10.,20.,6.);
        // // // // let v3: Vector3<f32> = Vector3::new(30.,50.,14.);

        // let p0 = GraphicalPlane::new(v1,v0,0xFFCC88);

        // // // // // // let p1 = Plane::new(v2,v3,0x3FDD88);

        // elements.push(Box::new(p0));
        // elements.push(Box::new(p1));


        //https://www.thkp.co/blog/2020/2/5/rendering-3d-from-scratch-chapter-7-the-depth-buffer
        //https://www.gabrielgambetta.com/computer-graphics-from-scratch/03-light.html
        
        // let teapot = Imported::new("../../3d_models/teapot.obj",10.,0.,0.,0.);
        
        // elements.push(Box::new(teapot));

        // let sphere = Imported::new("../../3d_models/sphere/source/Archive/sphere.obj",1.,15.,0.,0.);
        // elements.push(Box::new(sphere));

        // let g: f32 = 1.22/2.;
        // let x_pt = g.tan();


        // let cube = Cube::new(x_pt,0.,-60.,50., 0xFFFFFF);
        // elements.push(Box::new(cube));

        // let cube = Cube::new(-x_pt,0.,-1.,50., 0x0FFF0F);

        // elements.push(Box::new(cube));

        // let cube = Cube::new(-40.,-20.,3.,40., 0xFFFFFF);
        // elements.push(Box::new(cube));
        
        // let cube = Cube::new(20.,-20.,3.,40., 0xFFFFFF);
        // elements.push(Box::new(cube));
        
        let mut running_total = 0;

        let mut idx_vec_running: Vec<i32> = Vec::new();    

        for mesh in &elements {
            idx_vec_running.push(running_total);
            running_total += mesh.num_vertices() as i32;
        }


        World {
            elements, lights, idx_vec_running,
        }
    }

}