use nalgebra::{Vector3, Vector4};

pub struct Face{
    pub vertex_ids: Vector4<i64>,
    pub normal: Vector3<f32>,
    pub is_three_points: bool
    
}

impl Face {
    pub fn new_three(v1_id:i64,v2_id:i64,v3_id:i64, v1: Vector3<f32>, v2: Vector3<f32>, v3: Vector3<f32>) -> Face{

        // assume 3 point face

        // going to use right hand cross product, meaning points should be counter clockwise oriented
        let v = v2 - v1;
        let u = v3 - v1;

        let normal = v.cross(&u).normalize();

        let is_three_points = true;

        Face {
            vertex_ids: Vector4::new(v1_id,v2_id,v3_id,-1),
            normal,
            is_three_points,
        }
    }

    pub fn new_four(v1_id:i64,v2_id:i64,v3_id:i64, v4_id: i64, v1: Vector3<f32>, v2: Vector3<f32>, v3: Vector3<f32>, v4: Vector3<f32>) -> Face{

        // assume 3 point face

        // going to use right hand cross product, meaning points should be counter clockwise oriented
        let v = v2 - v1;
        let u = v3 - v1;

        let normal = v.cross(&u).normalize();

        let is_three_points: bool = false;

        Face {
            vertex_ids: Vector4::new(v1_id,v2_id,v3_id,v4_id),
            normal,
            is_three_points,
        }
    }

    pub fn new_from_vec_three(vertices: &Vec<i64>, v1: Vector3<f32>, v2: Vector3<f32>, v3: Vector3<f32>) -> Face {
        assert_eq!(vertices.len(),3);

        let v = v2 - v1;
        let u = v3 - v1;

        let normal = v.cross(&u).normalize();
        let is_three_points: bool = true;


        Face {
            vertex_ids: Vector4::new(vertices[0],vertices[1],vertices[2], -1),
            normal,
            is_three_points,
        }
    }

    pub fn new_from_vec_four(vertices: &Vec<i64>, v1: Vector3<f32>, v2: Vector3<f32>, v3: Vector3<f32>, v4: Vector3<f32>) -> Face {
        assert_eq!(vertices.len(),4);

        let v = v2 - v1;
        let u = v3 - v1;

        let normal = v.cross(&u).normalize();
        let is_three_points: bool = true;


        Face {
            vertex_ids: Vector4::new(vertices[0],vertices[1],vertices[2], vertices[3]),
            normal,
            is_three_points,
        }
    }

}