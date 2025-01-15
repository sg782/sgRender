use nalgebra::Vector3;

pub struct Face{
    pub vertices: Vector3<i64>,
}

impl Face {
    pub fn new(v1:i64,v2:i64,v3:i64) -> Face{
        Face {
            vertices: Vector3::new(v1,v2,v3)
        }
    }
}