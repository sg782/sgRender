use crate::mesh::face::Face;
use crate::mesh::point::Point;
use crate::mesh::mesh::Mesh;
use nalgebra::{Vector2,Vector3};

pub struct GraphicalPlane {
    vertices: Vec<Point>,
    faces: Vec<Face>,
    bounding_box: Vector2<Vector3<f32>>,
    color: u32,
    num_vertices: usize,

}

impl Mesh for GraphicalPlane {
    fn vertices(&self) -> &Vec<Point> {
        &self.vertices
    }

    fn faces(&self) -> &Vec<Face> {
        &self.faces
    }

    fn bounding_box(&self) -> &nalgebra::Vector2<nalgebra::Vector3<f32>> {
        &self.bounding_box
    }

    fn color(&self) -> u32 {
        self.color
    }

    fn num_vertices(&self) -> usize {
        self.num_vertices
    }
}

impl GraphicalPlane{
    
    pub fn new(v0: Vector3<f32>, v1: Vector3<f32>, color: u32) -> GraphicalPlane{
        let mut faces: Vec<Face> = Vec::new();
        let mut vertices: Vec<Point> = Vec::new();

        // kinda hacky definition of which point are important. I dont care
        vertices.push(Point::new(v0.x,v0.y,v0.z));
        vertices.push(Point::new(v0.x,v1.y,v0.z));
        vertices.push(Point::new(v1.x,v0.y,v1.z));
        vertices.push(Point::new(v1.x,v1.y,v1.z));

        faces.push(Face::new(
            0,1,2, 
            vertices[0].as_vec3(), 
            vertices[1].as_vec3(), 
            vertices[2].as_vec3()
        ));
        faces.push(Face::new(
            1,2,3, 
            vertices[1].as_vec3(), 
            vertices[2].as_vec3(), 
            vertices[3].as_vec3()
        ));


        
        let bounding_box: Vector2<Vector3<f32>> = Vector2::new(Vector3::new(v0.x,v0.y,v0.z),Vector3::new(v1.x,v1.y,v1.z));
        
        let num_vertices = vertices.len();


        GraphicalPlane {
            vertices,
            faces,
            bounding_box,
            color,
            num_vertices,
        }
        
    }
}

