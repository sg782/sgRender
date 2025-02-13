use nalgebra::Vector3;

pub struct Face{
    pub vertex_ids: Vector3<i64>,
    pub normal: Vector3<f32>
    
}

impl Face {
    pub fn new(v1_id:i64,v2_id:i64,v3_id:i64, v1: Vector3<f32>, v2: Vector3<f32>, v3: Vector3<f32>) -> Face{

        // going to use right hand cross product, meaning points should be counter clockwise oriented
        let v = v2 - v1;
        let u = v3 - v1;

        let normal = v.cross(&u).normalize();

        Face {
            vertex_ids: Vector3::new(v1_id,v2_id,v3_id),
            normal,
        }
    }

    pub fn new_from_vec(vertices: &Vec<i64>, v1: Vector3<f32>, v2: Vector3<f32>, v3: Vector3<f32>) -> Face {
        assert_eq!(vertices.len(),3);

        let v = v2 - v1;
        let u = v3 - v1;

        let normal = v.cross(&u).normalize();


        Face {
            vertex_ids: Vector3::new(vertices[0],vertices[1],vertices[2]),
            normal,
        }
    }

}