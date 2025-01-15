use crate::mesh::face::Face;
use crate::mesh::point::Point;

pub trait Mesh{
    fn vertices(&self) -> &Vec<Point>;
    fn faces(&self) -> &Vec<Face>;
}
