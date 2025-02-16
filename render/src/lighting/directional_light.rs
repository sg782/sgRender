use nalgebra::Vector3;

pub struct DirectionalLight {
    direction: Vector3<f32>,
    intensity: f32,
}

impl DirectionalLight {
    pub fn new(direction: Vector3<f32>, intensity: f32) -> DirectionalLight{
        DirectionalLight {
            direction,
            intensity
        }
    }
}