use nalgebra::Vector4;
pub struct Point{
    pub position: Vector4<f32>,
}

impl Point{
    pub fn new(x:f32, y:f32, z:f32) -> Point{
        Point{
            position: Vector4::new(x,y,z,1.),
        }
    }

    pub fn new_from_vec(coords: Vec<f32>) -> Point {
        assert_eq!(coords.len(),3);

        Point {
            position: Vector4::new(coords[0],coords[1],coords[2],1.)
        }
    }
}

