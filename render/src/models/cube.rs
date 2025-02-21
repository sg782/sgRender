use crate::mesh::face::Face;
use crate::mesh::point::Point;
use crate::mesh::mesh::Mesh;
use nalgebra::{Vector2,Vector3};

pub struct Cube {
    vertices: Vec<Point>,
    faces: Vec<Face>,
    bounding_box: Vector2<Vector3<f32>>,
    color: u32,
    num_vertices: usize,

}

impl Mesh for Cube {
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

impl Cube{
    
    pub fn new(x: f32, y:f32, z:f32, side_length:f32, color: u32) -> Cube{
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

        let bounding_box: Vector2<Vector3<f32>> = Vector2::new(Vector3::new(x,y,z),Vector3::new(x+l,y+l,z+l));
        
        let num_vertices = vertices.len();


        // Define the 12 triangular faces of the cube, carefully constructed for normal vectors to face outward

        // Bottom face (0, 1, 2, 3)


        // left (right)
        faces.push(Face::new_three(
            0,1,3, 
            vertices[0].as_vec3(), 
            vertices[1].as_vec3(), 
            vertices[3].as_vec3()
        ));
        faces.push(Face::new_three(
            0,3,2, 
            vertices[0].as_vec3(), 
            vertices[3].as_vec3(), 
            vertices[2].as_vec3()
        ));
        

        // top (back)
        faces.push(Face::new_three(
            1,5,7, 
            vertices[1].as_vec3(), 
            vertices[5].as_vec3(), 
            vertices[7].as_vec3()
        ));
        faces.push(Face::new_three(
            1,7,3, 
            vertices[1].as_vec3(), 
            vertices[7].as_vec3(), 
            vertices[3].as_vec3()
        ));
        

        // bottom (front)
        faces.push(Face::new_three(
            0,2,4, 
            vertices[0].as_vec3(), 
            vertices[2].as_vec3(), 
            vertices[4].as_vec3()
        ));
        faces.push(Face::new_three(
            2,6,4, 
            vertices[2].as_vec3(), 
            vertices[6].as_vec3(), 
            vertices[4].as_vec3()
        ));

        
        //right (left)
        faces.push(Face::new_three(
            4,6,5, 
            vertices[4].as_vec3(), 
            vertices[6].as_vec3(), 
            vertices[5].as_vec3()
        ));
        faces.push(Face::new_three(
            5,6,7, 
            vertices[5].as_vec3(), 
            vertices[6].as_vec3(), 
            vertices[7].as_vec3()
        ));



        //front (bottom)
        faces.push(Face::new_three(
            2,3,7, 
            vertices[2].as_vec3(), 
            vertices[3].as_vec3(), 
            vertices[7].as_vec3()
        ));
        faces.push(Face::new_three(
            2,7,6, 
            vertices[2].as_vec3(), 
            vertices[7].as_vec3(), 
            vertices[6].as_vec3()
        ));

        
        //back (top)
        faces.push(Face::new_three(
            0,5,1, 
            vertices[0].as_vec3(), 
            vertices[5].as_vec3(), 
            vertices[1].as_vec3()
        ));
        faces.push(Face::new_three(
            0,4,5, 
            vertices[0].as_vec3(), 
            vertices[4].as_vec3(), 
            vertices[5].as_vec3()
        ));



        /*
        Faces in counterclockwise order (opposite to as written above)
        
         */
           // left (right)

           /*
        faces.push(Face::new_three(
            0,3,1, 
            vertices[0].as_vec3(), 
            vertices[3].as_vec3(), 
            vertices[1].as_vec3()
        ));
        faces.push(Face::new_three(
            0,2,3, 
            vertices[0].as_vec3(), 
            vertices[2].as_vec3(), 
            vertices[3].as_vec3()
        ));
        

        // top (back)
        faces.push(Face::new_three(
            1,7,5, 
            vertices[1].as_vec3(), 
            vertices[7].as_vec3(), 
            vertices[5].as_vec3()
        ));
        faces.push(Face::new_three(
            1,3,7, 
            vertices[1].as_vec3(), 
            vertices[3].as_vec3(), 
            vertices[7].as_vec3()
        ));
        

        // bottom (front)
        faces.push(Face::new_three(
            0,4,2, 
            vertices[0].as_vec3(), 
            vertices[4].as_vec3(), 
            vertices[2].as_vec3()
        ));
        faces.push(Face::new_three(
            2,4,6, 
            vertices[2].as_vec3(), 
            vertices[4].as_vec3(), 
            vertices[6].as_vec3()
        ));

        
        //right (left)
        faces.push(Face::new_three(
            4,5,6, 
            vertices[4].as_vec3(), 
            vertices[5].as_vec3(), 
            vertices[6].as_vec3()
        ));
        faces.push(Face::new_three(
            5,7,6, 
            vertices[5].as_vec3(), 
            vertices[7].as_vec3(), 
            vertices[6].as_vec3()
        ));



        //front (bottom)
        faces.push(Face::new_three(
            2,7,3, 
            vertices[2].as_vec3(), 
            vertices[7].as_vec3(), 
            vertices[3].as_vec3()
        ));
        faces.push(Face::new_three(
            2,6,7, 
            vertices[2].as_vec3(), 
            vertices[6].as_vec3(), 
            vertices[7].as_vec3()
        ));

        
        //back (top)
        faces.push(Face::new_three(
            0,1,5, 
            vertices[0].as_vec3(), 
            vertices[1].as_vec3(), 
            vertices[5].as_vec3()
        ));
        faces.push(Face::new_three(
            0,5,4, 
            vertices[0].as_vec3(), 
            vertices[5].as_vec3(), 
            vertices[4].as_vec3()
        ));
        
        
        */
         

        

        Cube {
            vertices,
            faces,
            bounding_box,
            color,
            num_vertices,
        }
        
    }
}


/*

        faces.push(Face::new_three(0,1,3));
        faces.push(Face::new_three(0,2, 3));

        // top
        faces.push(Face::new_three(1, 5, 7));
        faces.push(Face::new_three(1, 3, 7));

        // bottom
        faces.push(Face::new_three(0, 2, 4));
        faces.push(Face::new_three(2, 4, 6));

        //right
        faces.push(Face::new_three(4, 5, 6));
        faces.push(Face::new_three(5, 6, 7));

        //front
        faces.push(Face::new_three(2, 3, 7));
        faces.push(Face::new_three(2, 6, 7));

        //back
        faces.push(Face::new_three(0, 1, 5));
        faces.push(Face::new_three(0, 4, 5)); */
