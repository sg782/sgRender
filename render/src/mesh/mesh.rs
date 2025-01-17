use crate::mesh::face::Face;
use crate::mesh::point::Point;

pub trait Mesh: Send + Sync{
    fn vertices(&self) -> &Vec<Point>;
    fn faces(&self) -> &Vec<Face>;
}
