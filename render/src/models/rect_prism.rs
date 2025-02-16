use crate::mesh::face::Face;
use crate::mesh::point::Point;
use crate::mesh::mesh::Mesh;
use nalgebra::{Vector2,Vector3};

pub struct RectPrism {
    vertices: Vec<Point>,
    faces: Vec<Face>,
    bounding_box: Vector2<Vector3<f32>>,
    color: u32,
    num_vertices: usize,

}

impl Mesh for RectPrism {
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

impl RectPrism{
    
    pub fn new(x: f32, y:f32, z:f32, x_length: f32, y_length: f32, z_length: f32, color: u32) -> RectPrism{
        let mut faces: Vec<Face> = Vec::new();
        let mut vertices: Vec<Point> = Vec::new();
        
        
        
        vertices.push(Point::new(x,y,z)); // 0
        vertices.push(Point::new(x,y,z+z_length)); // 1
        vertices.push(Point::new(x,y+y_length,z)); // 2
        vertices.push(Point::new(x,y+y_length,z+z_length)); // 3

        vertices.push(Point::new(x+x_length,y,z)); // 4
        vertices.push(Point::new(x+x_length,y,z+z_length)); // 5
        vertices.push(Point::new(x+x_length,y+y_length,z)); // 6
        vertices.push(Point::new(x+x_length,y+y_length,z+z_length)); // 7

        let bounding_box: Vector2<Vector3<f32>> = Vector2::new(Vector3::new(x,y,z),Vector3::new(x+x_length,y+y_length,z+z_length));
        
        let num_vertices = vertices.len();


        // Define the 12 triangular faces of the RectPrism, carefully constructed for normal vectors to face outward

        // Bottom face (0, 1, 2, 3)


        // left (right)
        faces.push(Face::new(
            0,1,3, 
            vertices[0].as_vec3(), 
            vertices[1].as_vec3(), 
            vertices[3].as_vec3()
        ));
        faces.push(Face::new(
            0,3,2, 
            vertices[0].as_vec3(), 
            vertices[3].as_vec3(), 
            vertices[2].as_vec3()
        ));
        

        // top (back)
        faces.push(Face::new(
            1,5,7, 
            vertices[1].as_vec3(), 
            vertices[5].as_vec3(), 
            vertices[7].as_vec3()
        ));
        faces.push(Face::new(
            1,7,3, 
            vertices[1].as_vec3(), 
            vertices[7].as_vec3(), 
            vertices[3].as_vec3()
        ));
        

        // bottom (front)
        faces.push(Face::new(
            0,2,4, 
            vertices[0].as_vec3(), 
            vertices[2].as_vec3(), 
            vertices[4].as_vec3()
        ));
        faces.push(Face::new(
            2,6,4, 
            vertices[2].as_vec3(), 
            vertices[6].as_vec3(), 
            vertices[4].as_vec3()
        ));

        
        //right (left)
        faces.push(Face::new(
            4,6,5, 
            vertices[4].as_vec3(), 
            vertices[6].as_vec3(), 
            vertices[5].as_vec3()
        ));
        faces.push(Face::new(
            5,6,7, 
            vertices[5].as_vec3(), 
            vertices[6].as_vec3(), 
            vertices[7].as_vec3()
        ));



        //front (bottom)
        faces.push(Face::new(
            2,3,7, 
            vertices[2].as_vec3(), 
            vertices[3].as_vec3(), 
            vertices[7].as_vec3()
        ));
        faces.push(Face::new(
            2,7,6, 
            vertices[2].as_vec3(), 
            vertices[7].as_vec3(), 
            vertices[6].as_vec3()
        ));

        
        //back (top)
        faces.push(Face::new(
            0,5,1, 
            vertices[0].as_vec3(), 
            vertices[5].as_vec3(), 
            vertices[1].as_vec3()
        ));
        faces.push(Face::new(
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
        faces.push(Face::new(
            0,3,1, 
            vertices[0].as_vec3(), 
            vertices[3].as_vec3(), 
            vertices[1].as_vec3()
        ));
        faces.push(Face::new(
            0,2,3, 
            vertices[0].as_vec3(), 
            vertices[2].as_vec3(), 
            vertices[3].as_vec3()
        ));
        

        // top (back)
        faces.push(Face::new(
            1,7,5, 
            vertices[1].as_vec3(), 
            vertices[7].as_vec3(), 
            vertices[5].as_vec3()
        ));
        faces.push(Face::new(
            1,3,7, 
            vertices[1].as_vec3(), 
            vertices[3].as_vec3(), 
            vertices[7].as_vec3()
        ));
        

        // bottom (front)
        faces.push(Face::new(
            0,4,2, 
            vertices[0].as_vec3(), 
            vertices[4].as_vec3(), 
            vertices[2].as_vec3()
        ));
        faces.push(Face::new(
            2,4,6, 
            vertices[2].as_vec3(), 
            vertices[4].as_vec3(), 
            vertices[6].as_vec3()
        ));

        
        //right (left)
        faces.push(Face::new(
            4,5,6, 
            vertices[4].as_vec3(), 
            vertices[5].as_vec3(), 
            vertices[6].as_vec3()
        ));
        faces.push(Face::new(
            5,7,6, 
            vertices[5].as_vec3(), 
            vertices[7].as_vec3(), 
            vertices[6].as_vec3()
        ));



        //front (bottom)
        faces.push(Face::new(
            2,7,3, 
            vertices[2].as_vec3(), 
            vertices[7].as_vec3(), 
            vertices[3].as_vec3()
        ));
        faces.push(Face::new(
            2,6,7, 
            vertices[2].as_vec3(), 
            vertices[6].as_vec3(), 
            vertices[7].as_vec3()
        ));

        
        //back (top)
        faces.push(Face::new(
            0,1,5, 
            vertices[0].as_vec3(), 
            vertices[1].as_vec3(), 
            vertices[5].as_vec3()
        ));
        faces.push(Face::new(
            0,5,4, 
            vertices[0].as_vec3(), 
            vertices[5].as_vec3(), 
            vertices[4].as_vec3()
        ));
        
        
        */
         

        

        RectPrism {
            vertices,
            faces,
            bounding_box,
            color,
            num_vertices,
        }
        
    }
}


/*

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
        faces.push(Face::new(0, 4, 5)); */
