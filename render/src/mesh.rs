use crate::point::Point;
use crate::face::Face;


pub struct Mesh{
    pub points: Vec<Point>,
    pub faces: Vec<Face>,
}

impl Mesh {
    pub fn new(points: Vec<Point>, faces: Vec<Face>) -> Mesh{
        Mesh {
            points,
            faces
        }
    }
}