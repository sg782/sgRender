use crate::mesh::face::Face;
use crate::mesh::point::Point;
use crate::mesh::mesh::Mesh;
use crate::models::cube::Cube;

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
        let mut elements: Vec<Box<dyn Mesh>> = Vec::new();

        // test with a row of cubes
        let side_length = 2.;
        for i in -5..5{
            let idx = i as f64;
            let cube = Cube::new(side_length * idx,-10.,-10.,side_length);
            elements.push(Box::new(cube));

        }
        

        World {
            height, width, depth, elements
        }
    }

}