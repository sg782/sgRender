use crate::mesh::face::Face;
use crate::mesh::point::Point;
use nalgebra::Vector2;
use nalgebra::Vector3;



pub trait Mesh: Send + Sync{
    fn vertices(&self) -> &Vec<Point>;
    fn faces(&self) -> &Vec<Face>;
    fn bounding_box(&self) -> &Vector2<Vector3<f64>>;
    fn color(&self) -> u32;
    // fn get cubic outline, only render if cubic outline is inside
}
