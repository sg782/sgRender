use nalgebra::Vector3;
// for defining the frustum bounding planes
pub struct Plane {
    pub normal: Vector3<f32>,
    pub distance: f32,
}


/*
RIGHT HAND coord system

+Z is out of screen (at start)
+X is right
+Y is up

*/

impl Plane {
    //https://www.lighthouse3d.com/tutorials/maths/plane/
    pub fn from_points(p0:Vector3<f32>, p1: Vector3<f32>, p2: Vector3<f32>) -> Plane{
        let v = p1 - p0;
        let u = p2 - p0;
        let n = v.cross(&u).normalize();

        let distance = - (n.dot(&p0));

        Plane {
            normal: n,
            distance,
        }
    }

    // pub fn is_inside(&self, p: Vector3<f32> ) -> bool{

    //     self.normal.dot(&p) - self.distance <=0.0
    // }


}