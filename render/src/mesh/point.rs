use nalgebra::Vector4;
pub struct Point{
    pub position: Vector4<f64>,
}

impl Point{
    pub fn new(x:f64, y:f64, z:f64) -> Point{
        Point{
            position: Vector4::new(x,y,z,1.),
        }
    }

    pub fn new_from_vec(coords: Vec<f64>) -> Point {
        assert_eq!(coords.len(),3);

        Point {
            position: Vector4::new(coords[0],coords[1],coords[2],1.)
        }
    }
}

