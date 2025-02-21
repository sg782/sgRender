use nalgebra::{Vector3, Vector4};

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

    pub fn as_vec_4(&self) -> Vector4<f32> {
        Vector4::new(self.pos.x, self.pos.y, self.pos.z,self.intensity)
    }
}