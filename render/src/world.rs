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

        //test with a row of cubes
        let side_length = 5.9;
        let amount_wide = 45;
        let amount_high = 45;
        for i in -amount_wide..amount_wide{
            for j in -amount_high..amount_high{
                    let idx = i as f64;
                    let jdx = j as f64;
                    let cube = Cube::new(side_length * idx,side_length*jdx,-10.,side_length);
                    elements.push(Box::new(cube));
            }
        }

        
        let teapot = Imported::new("../../3d_models/teapot.obj",100.,0.,0.,0.);
        // print!("Teapot: {}", teapot.bounding_box());

        // elements.push(Box::new(teapot));


        let cube = Cube::new(-5.,-20.,1.,100.,);
        print!("Teapot: {}", cube.bounding_box());

        elements.push(Box::new(cube));


        
        World {
            height, width, depth, elements
        }
    }

}