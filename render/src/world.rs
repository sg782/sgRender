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

        let mut faces: Vec<Face> = Vec::new();
        let mut points: Vec<Point> = Vec::new();

        // points.push(Point::new(-100.,-100.,-10.));
        // points.push(Point::new(-100.,300.,-10.));
        // points.push(Point::new(100.,-100.,-10.));

        // faces.push(Face::new(0,1,2));




        // random test data

        // generate a cube

        // Define the 8 vertices of the cube
        
        
        
        let l = 10.0;
        points.push(Point::new(-l,-l,-l)); // 0
        points.push(Point::new(-l,-l,l)); // 1
        points.push(Point::new(-l,l,-l)); // 2
        points.push(Point::new(-l,l,l)); // 3

        points.push(Point::new(l,-l,-l)); // 4
        points.push(Point::new(l,-l,l)); // 5
        points.push(Point::new(l,l,-l)); // 6
        points.push(Point::new(l,l,l)); // 7


        // Define the 12 triangular faces of the cube

        // Bottom face (0, 1, 2, 3)

        // left
        faces.push(Face::new(0,1,3));
        faces.push(Face::new(0,2, 3));

        // top
        faces.push(Face::new(1, 5, 7));
        faces.push(Face::new(1, 3, 7));

        // bottom
        faces.push(Face::new(0, 2, 4));
        faces.push(Face::new(2, 4, 6));

        //right
        faces.push(Face::new(4, 5, 6));
        faces.push(Face::new(5, 6, 7));

        //front
        faces.push(Face::new(2, 3, 7));
        faces.push(Face::new(2, 6, 7));

        //back
        faces.push(Face::new(0, 1, 5));
        faces.push(Face::new(5, 4, 5));
        

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