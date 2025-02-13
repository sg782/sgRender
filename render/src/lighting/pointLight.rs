use nalgebra::Vector3;

pub struct PointLight {
    pos: Vector3<f32>,
    intensity: f32,
}

impl PointLight {
    pub fn new(pos: Vector3<f32>, intensity: f32) -> PointLight{
        PointLight {
            pos,
            intensity
        }
    }
}