use crate::world::World;
use crate::view::View;
use nalgebra::Vector2;
use rayon::prelude::*;
use crate::mesh::mesh::Mesh;

use crate::primitives::line::Line;

use nalgebra::Matrix4;
use nalgebra::Vector4;

use crate::primitives::triangle::Triangle;
use nalgebra::Vector3;

use num_cpus;

/*
RIGHT HANDED COORDINATE SYSTEM:

+x = RIGHT
+y = UP
+z = FORWARD

*/

pub struct Renderer{
    pub world: World,
    pub view: View,
}

impl Renderer {
    pub fn new(world: World, view: View) -> Renderer{
        Renderer{
            world,
            view,
        }
    }

    pub fn clip_at_edge(x1: &mut f64, y1: &mut f64, x2: &mut f64, y2: &mut f64, screen_width: i64, screen_height: i64){

        let width = screen_width as f64;
        let height =screen_height as f64;

        if(*x1 <=0. && *x2 <=0.){
            return;
        }
        if(*x1 >=width-1. && *x2>= width-1.){
            return;
        }

        if(*y1 <=0. && *y2 <= 0.){
            return;
        }
        if(*y1 >=height-1. && *y2>= height-1.){
            return;
        }


        // x
        if *x1 < 0. {
            let scaling = *x2 / (*x2-*x1);


            let dy = *y2 - *y1;

            *x1 = 0.;
            *y1 = *y2 - (scaling * dy);
        } else if *x1 >= width {
            let scaling = (width - *x2) / (*x1 - *x2);
            let dy = *y2 - * y1;
            *x1 = width-1.;
            *y1 = *y2 - (scaling * dy);
        }

        if *x2 <0. {
            let scaling = *x1 / (*x1-*x2);
            let dy = *y1 - *y2;

            *x2 = 0.;
            *y2 = *y1 - (scaling * dy);
        }else if *x2 >= width {
            let scaling = (width - *x1) / (*x2 - *x1);
            let dy = *y1 - * y2;
            *x2 = width-1.;
            *y2 = *y1 - (scaling * dy);
        }


        // y
        if *y1 < 0. {
            let scaling = *y2 / (*y2-*y1);
            let dx = *x2 - *x1;

            *y1 = 0.;
            *x1 = *x2 - (scaling * dx);
        }else if *y1 >= height {
            let scaling = (height - *y2) / (*y1 - *y2);
            let dx = *x2 - * x1;
            *y1 = height-1.;
            *x1 = *x2 - (scaling * dx);
        }

        if *y2 <0. {
            let scaling = *y1 / (*y1-*y2);
            let dx = *x1 - *x2;

            *y2 = 0.;
            *x2 = *x1 - (scaling * dx);
        } else if *y2 >= height {
            let scaling = (height - *y1) / (*y2 - *y1);
            let dx = *x1 - * x2;
            *y2 = height-1.;
            *x2 = *x1 - (scaling * dx);
        }



    }

    pub fn calculate_transformation(&self) -> Matrix4<f64>{

        let near = self.view.near;

        let far = self.view.far; // operates as a max render distance


        let fov = self.view.fov;  // ~70 degrees
        let depth = far-near;

        let alpha = self.view.roll;
        let beta = self.view.pitch;
        let gamma: f64 = self.view.yaw;

        let c1 = -(far+near) / depth;
        let c2 = -2.*far*near / depth; 

        // aspect ratio
        let a = self.view.aspect_ratio;

        let tan_fov = ((fov/2.) as f64).tan();

        let view_translation = Matrix4::new(
            1., 0., 0., self.view.x, 
            0., 1., 0., self.view.y, 
            0., 0., 1., self.view.z, 
            0., 0., 0., 1.,
        );

        let x_rotation = Matrix4::new(
            1., 0., 0., 0., 
            0., alpha.cos(), -(alpha.sin()), 0.,
            0., alpha.sin(), alpha.cos(), 0., 
            0., 0., 0., 1.,
        );

        let y_rotation = Matrix4::new(
            beta.cos(), 0., beta.sin(), 0.,
            0., 1., 0., 0., 
            -(beta.sin()), 0., beta.cos(), 0., 
            0., 0., 0., 1.,
        );

        let z_rotation = Matrix4::new(
            gamma.cos(), -(gamma.sin()), 0., 0.,
            gamma.sin(), gamma.cos(), 0., 0., 
            0., 0., 1., 0.,
            0., 0., 0., 1.,
        );

        let perspective_transformation = Matrix4::new(
            1./(a*tan_fov), 0., 0., 0., 
            0., 1./tan_fov, 0., 0.,
            0., 0., c1, c2,
            0., 0., -1., 0.,
        );

        let full_transformation = perspective_transformation * z_rotation * y_rotation * x_rotation * view_translation;

        return full_transformation;
    }

    pub fn render(&self, buffer: &mut Vec<u32>, use_wireframe: bool, screen_width: i64, screen_height: i64){
        //buffer.fill(0x000000);
        buffer.fill(0x87CEFA);

        let full_transformation = self.calculate_transformation();

        let num_cores = num_cpus::get();
        let fragment_size = (&self.world.elements.len() / num_cores) + 1;

        /*
        multithreading

        I tried to do it manually (bad idea)
        using multithreading library 'rayon' now. it does not seem to help much
        
        Only 16 threads, i would like to delve into gpu threading maybe

         */

        // draw wirefram instead of faces
        if use_wireframe {
            let collected_data: Vec<Vec<Vec<Vector4<f64>>>> = 
            self.world.elements.par_chunks(fragment_size).map(
                |chunk| {
    
                    let mut thread_data: Vec<Vec<Vector4<f64>>> = Vec::new();
                    for mesh in chunk.iter() {
                        thread_data.push(self.calculate_mesh_wireframe(mesh, full_transformation, screen_width, screen_height));
                    }
                    thread_data
                }
            ).collect();
    
            for i in collected_data.iter(){
                for j in i.iter(){
                    for k in j.iter() {
                        let line = Line::new(k[0],k[1],k[2],k[3],1.,0xFF0000);
                        line.draw(buffer, screen_width, screen_height);
                    }
                }
            }

            return;
        }

        let collected_data: Vec<Vec<Vec<Vector3<Vector2<f64>>>>> = 
        self.world.elements.par_chunks(fragment_size).map(
            |chunk| {

                let mut thread_data:  Vec<Vec<Vector3<Vector2<f64>>>> = Vec::new();
                for mesh in chunk.iter() {
                    thread_data.push(self.calculate_mesh_faces(mesh, full_transformation, screen_width, screen_height));
                }
                thread_data
            }
        ).collect();

        for i in collected_data.iter(){
            for j in i.iter(){
                for k in j.iter() {
                    let triangle = Triangle::from_vec_list(k, 0xFFFFFF);
                    triangle.draw(buffer, screen_width, screen_height);
                }
            }
        }


        // draw faces (default behavior)


    }


    fn calculate_mesh_wireframe(&self, mesh: &Box<dyn Mesh>, full_transformation: Matrix4<f64>, screen_width: i64, screen_height: i64) -> Vec<Vector4<f64>>{
        let width = screen_width as f64;
        let height = screen_height as f64;


        let mut mesh_data: Vec<Vector4<f64>> = Vec::new();


        // if not in view, dont render
        if !self.view.in_view(mesh){
            return mesh_data;
        }
        
        let mut transformed_vertices: Vec<Vector4<f64>> = Vec::new();
        let mut w_vals: Vec<f64> = Vec::new();
        for point in mesh.vertices(){
            let transformed_vertex = full_transformation * point.position;
            w_vals.push(transformed_vertex[3]);
            transformed_vertices.push(transformed_vertex / transformed_vertex[3]);
            
        } 

        // render each face
        for face in mesh.faces(){

            // rendering lines
            for i in 0..3 {

                let id_a = face.vertices[i] as usize;
                let id_b = face.vertices[(i+1)%3] as usize;

                if w_vals[id_a] > -self.view.near || w_vals[id_b] > -self.view.near {
                    continue;
                }

                let a_position_prime = transformed_vertices[id_a];
                let b_position_prime = transformed_vertices[id_b];

                // scale back to device coordinates
                // can remove some parameter passing by scaling AFTER clipping, idc rn
                let mut x1_prime = width * (a_position_prime[0] +1.)/2.;
                let mut y1_prime = height * (1.-a_position_prime[1])/2.;
                let mut x2_prime = width * (b_position_prime[0] +1.)/2.;
                let mut y2_prime = height * (1.-b_position_prime[1])/2.;

                Renderer::clip_at_edge(&mut x1_prime,&mut y1_prime,&mut x2_prime,&mut y2_prime,screen_width,screen_height);
                mesh_data.push(Vector4::new(x1_prime,y1_prime,x2_prime,y2_prime));
            }

            // render triangles
        }

        mesh_data
    }

    fn calculate_mesh_faces(&self, mesh: &Box<dyn Mesh>, full_transformation: Matrix4<f64>, screen_width: i64, screen_height: i64) ->  Vec<Vector3<Vector2<f64>>>{
        let width = screen_width as f64;
        let height = screen_height as f64;


        let mut mesh_data: Vec<Vector3<Vector2<f64>>> = Vec::new();


        // if not in view, dont render
        if !self.view.in_view(mesh){
            return mesh_data;
        }
        
        let mut transformed_vertices: Vec<Vector4<f64>> = Vec::new();
        let mut w_vals: Vec<f64> = Vec::new();
        for point in mesh.vertices(){
            let transformed_vertex = full_transformation * point.position;
            w_vals.push(transformed_vertex[3]);
            transformed_vertices.push(transformed_vertex / transformed_vertex[3]);
            
        } 

        // render each face
        for face in mesh.faces(){

            let mut face_data: Vector3<Vector2<f64>> = Vector3::new(
                Vector2::new(0.,0.,),
                Vector2::new(0.,0.,),
                Vector2::new(0.,0.,),
            );
            // rendering lines
            for i in 0..3 {

                let id_a = face.vertices[i] as usize;

                // if w_vals[id_a] > -self.view.near || w_vals[id_b] > -self.view.near {
                //     continue;
                // }

                let a_position_prime = transformed_vertices[id_a];

                // scale back to device coordinates
                // can remove some parameter passing by scaling AFTER clipping, idc rn
                let x1_prime = width * (a_position_prime[0] +1.)/2.;
                let y1_prime = height * (1.-a_position_prime[1])/2.;

                face_data[i] =Vector2::new(x1_prime,y1_prime);
            }

            mesh_data.push(face_data);

            // render triangles
        }

        mesh_data
    }
}

