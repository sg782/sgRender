use crate::mesh::face::Face;
use crate::mesh::point::Point;
use crate::mesh::mesh::Mesh;
use nalgebra::{Vector2,Vector3};

pub struct Cube {
    vertices: Vec<Point>,
    faces: Vec<Face>,
    bounding_box: Vector2<Vector3<f64>>,
    color: u32,

}

impl Mesh for Cube {
    fn vertices(&self) -> &Vec<Point> {
        &self.vertices
    }

    fn faces(&self) -> &Vec<Face> {
        &self.faces
    }

    fn bounding_box(&self) -> &nalgebra::Vector2<nalgebra::Vector3<f64>> {
        &self.bounding_box
    }

    fn color(&self) -> u32 {
        self.color
    }
}

impl Cube{
    
    pub fn new(x: f64, y:f64, z:f64, side_length:f64, color: u32) -> Cube{
        let mut faces: Vec<Face> = Vec::new();
        let mut vertices: Vec<Point> = Vec::new();
        
        
        let l = side_length;
        vertices.push(Point::new(x,y,z)); // 0
        vertices.push(Point::new(x,y,z+l)); // 1
        vertices.push(Point::new(x,y+l,z)); // 2
        vertices.push(Point::new(x,y+l,z+l)); // 3

        vertices.push(Point::new(x+l,y,z)); // 4
        vertices.push(Point::new(x+l,y,z+l)); // 5
        vertices.push(Point::new(x+l,y+l,z)); // 6
        vertices.push(Point::new(x+l,y+l,z+l)); // 7

        let bounding_box: Vector2<Vector3<f64>> = Vector2::new(Vector3::new(x,y,z),Vector3::new(x+l,y+l,z+l));
        


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
        faces.push(Face::new(0, 4, 5));

        Cube {
            vertices,
            faces,
            bounding_box,
            color,
        }
        
    }
}

