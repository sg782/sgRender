use nalgebra::Vector3;
pub struct Point{
    pub position: Vector3<f64>,
}

impl Point{
    pub fn new(x:f64, y:f64, z:f64) -> Point{
        Point{
            position: Vector3::new(x,y,z),
        }
    }
}

