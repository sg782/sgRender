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
        let face = Face::new(0,1,2);
        let point1 = Point::new(50.,100.,30.);
        let point2 = Point::new(100.,100.,30.);
        let point3 = Point::new(65.,250.,30.);

        let mut points: Vec<Point> = Vec::new();
        points.push(point1);
        points.push(point2);
        points.push(point3);

        let mut faces: Vec<Face> = Vec::new();
        faces.push(face);


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