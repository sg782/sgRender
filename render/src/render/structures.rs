use bytemuck::{Pod, Zeroable};


#[repr(C)]
#[derive(Default, Copy, Clone, Pod, Zeroable)]
pub struct Transformation {
    pub transform_matrix: [[f32; 4]; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct BoundingPoint {
    pub min: [f32 ; 3],
    pub _pad1: f32,
    pub max: [f32 ; 3],
    pub _pad2: f32,
}

#[repr(C)] 
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
pub struct FrustumFaces {
    pub faces: [[ f32; 4];6], // normal: Vec3<f32>, and Distance: f32
}

#[repr(C)] 
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
pub struct PushConstants {
    pub a: f32,
    pub b: f32,
}

#[repr(C)] 
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
pub struct PushConstantsB {
    pub a: f32,
    pub b: f32,
    pub c: f32,
}

#[repr(C)] 
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
pub struct PushConstantsSortVertices {
    pub b: u32,
    pub c: u32,
    pub a: u32,
}

#[repr(C)] 
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
pub struct PushConstantsC {
    pub screen_width: f32,
    pub screen_height: f32,
    pub heading_x: f32,
    pub heading_y: f32, 
    pub heading_z: f32,
    pub x: f32, 
    pub y: f32,
    pub z: f32,
}