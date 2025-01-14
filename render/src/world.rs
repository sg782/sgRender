use crate::mesh::Mesh;
use crate::face::Face;
use crate::point::Point;

pub struct World{
    pub height: i64,
    pub width: i64,
    pub depth: i64,
    pub elements: Vec<Mesh>,
}

impl World{

    pub fn new(
        height: i64,
        width: i64,
        depth: i64,
    ) -> World {
        let mut elements: Vec<Mesh> = Vec::new();


        // random test data

        // generate a cube

        // Define the 8 vertices of the cube
        let mut points: Vec<Point> = Vec::new();
        points.push(Point::new(-10.0, 0.0, 20.0)); // 0
        points.push(Point::new(200.0, 0.0, 20.0)); // 1
        points.push(Point::new(120.0, 50.0, 14.0)); // 2

        points.push(Point::new(175.0, -100.0, 17.0)); // 3

        // Define the 12 triangular faces of the cube
        let mut faces: Vec<Face> = Vec::new();

        // Bottom face (0, 1, 2, 3)
        faces.push(Face::new(0, 1, 2));

        faces.push(Face::new(0, 1, 3));
        faces.push(Face::new(0, 2, 3));
        faces.push(Face::new(1, 2, 3));


        let mesh = Mesh::new(points, faces);


        elements.push(mesh);

        World {
            height, width, depth, elements
        }
    }

    // renders as is from the pov of the 
    pub fn render(&self, buffer: &mut Vec<u32>,  screen_width: i64, screen_height: i64){
        buffer.fill(0);
    }



}