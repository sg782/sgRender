use crate::mesh::face::Face;
use crate::mesh::point::Point;
use crate::mesh::mesh::Mesh;
use crate::models::cube::Cube;
use std::sync::Arc;
use crate::models::imported::Imported;

pub struct World{
    pub height: i64,
    pub width: i64,
    pub depth: i64,
    pub elements: Vec<Box<dyn Mesh>>,
}

impl World{

    pub fn new(
        height: i64,
        width: i64,
        depth: i64,
    ) -> World {


        /*
        
        
        ADJUST 'amount_high' and 'amount_wide' for quick changes in the models.
        you can also make a custom shape of any kind if you implement it with 'mesh' traits
        
         */


        let mut elements: Vec<Box<dyn Mesh>> = Vec::new();

        // // test with a row of cubes
        // let side_length = 5.;
        // let amount_wide = 100;
        // let amount_high = 1;
        // for i in 0..amount_wide{
        //     for j in 0..amount_high{
        //             let idx = i as f64;
        //             let jdx = j as f64;
        //             let cube = Cube::new(side_length * idx,side_length*jdx,-10.,side_length);
        //             elements.push(Box::new(cube));
        //     }
        // }

        let teapot = Imported::new("../../3d_models/teapot.obj",100.);
        elements.push(Box::new(teapot));
        
        World {
            height, width, depth, elements
        }
    }

}