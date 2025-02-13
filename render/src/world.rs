
use nalgebra::Vector3;

use crate::mesh::mesh::Mesh;
use crate::models::cube::Cube;
use crate::models::graphical_plane::GraphicalPlane;

use crate::models::imported::Imported;

pub struct World{
    pub elements: Vec<Box<dyn Mesh>>,
    pub idx_vec_running: Vec<u32>,
}

impl World{

    pub fn new() -> World {


        /*
        ADJUST 'amount_high' and 'amount_wide' for quick changes in the models.
        you can also make a custom shape of any kind if you implement it with 'mesh' traits
         */


        let mut elements: Vec<Box<dyn Mesh>> = Vec::new();

        // //test with a row of cubes
        // let side_length = 4.;
        // let amount_wide = 10;
        // let amount_high = 10;
        // let amount_deep = 10;
        // let spacing = 4.;
        // let mut count: f32 = 0.;
        // for i in 0..amount_wide{
        //     for j in 0..amount_high{
        //         for k in 0..amount_deep{
        //             count +=1.;
        //             let idx = i as f32;
        //             let jdx = j as f32;
        //             let kdx = k as f32;
        //             let r = 4.* (4.1+kdx);
        //             //let cube = Cube::new(r * (count/300.).sin() ,r * (count/200.).cos(), count / 5. ,side_length, 0xA028 * count as u32);
        //             let cube = Cube::new(idx * (side_length + spacing),jdx * (side_length + spacing), kdx * (side_length + spacing), side_length-1., 0xC028 * count as u32);

        //             // let teapot = Imported::new("../../3d_models/teapot.obj",side_length,idx * side_length, jdx * side_length, kdx*side_length);

        //             // elements.push(Box::new(teapot));

        //             elements.push(Box::new(cube));

        //         }

        //     }
        // }

        //println!("NUM items; {}", count);

        // let v0: Vector3<f32> = Vector3::new(10.,20.,14.);
        // let v1: Vector3<f32> = Vector3::new(30.,50.,6.);

        // let v2: Vector3<f32> = Vector3::new(10.,20.,6.);
        // let v3: Vector3<f32> = Vector3::new(30.,50.,14.);

        // let p0 = Plane::new(v1,v0,0xFFCC88);

        // let p1 = Plane::new(v2,v3,0x3FDD88);

        // elements.push(Box::new(p0));
        // elements.push(Box::new(p1));


        //https://www.thkp.co/blog/2020/2/5/rendering-3d-from-scratch-chapter-7-the-depth-buffer
        //https://www.gabrielgambetta.com/computer-graphics-from-scratch/03-light.html
        
        let teapot = Imported::new("../../3d_models/teapot.obj",10.,0.,0.,0.);

        elements.push(Box::new(teapot));
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

        let mut idx_vec_running: Vec<u32> = Vec::new();    

        for mesh in &elements {
            idx_vec_running.push(running_total);
            running_total += mesh.num_vertices() as u32;
        }


        World {
            elements, idx_vec_running,
        }
    }

}