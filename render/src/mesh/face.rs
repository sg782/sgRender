use nalgebra::Vector3;

pub struct Face{
    pub vertex_ids: Vector3<i64>,
    
}

impl Face {
    pub fn new(v1:i64,v2:i64,v3:i64) -> Face{
        Face {
            vertex_ids: Vector3::new(v1,v2,v3)
        }
    }

    pub fn new_from_vec(vertices: Vec<i64>) -> Face {
        assert_eq!(vertices.len(),3);

        Face {
            vertex_ids: Vector3::new(vertices[0],vertices[1],vertices[2])
        }
    }

}