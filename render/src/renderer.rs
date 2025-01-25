use std::f32::INFINITY;

use crate::world::World;
use crate::view::View;
use nalgebra::Transform;
use nalgebra::Vector2;
use rayon::prelude::*;
use vulkano::descriptor_set;
use vulkano::descriptor_set::layout::DescriptorSetLayoutBinding;
use crate::mesh::mesh::Mesh;

use crate::primitives::line::Line;

use std::collections::BTreeMap;

use nalgebra::Matrix4;
use nalgebra::Vector4;

use crate::primitives::triangle::Triangle;

use nalgebra::Vector3;


use std::sync::{Arc, Mutex};

use std::fs;

use vulkano::VulkanLibrary;
use vulkano::instance::{Instance, InstanceCreateFlags, InstanceCreateInfo};
use vulkano::device::QueueFlags;

use vulkano::device::{Device, DeviceCreateInfo, QueueCreateInfo};

use vulkano::memory::allocator::StandardMemoryAllocator;


use vulkano::descriptor_set::layout::{DescriptorSetLayout,DescriptorSetLayoutCreateInfo,DescriptorType};
use vulkano::pipeline::layout::{PipelineLayout, PipelineLayoutCreateInfo};
use vulkano::shader::ShaderStages;

use vulkano::pipeline::layout::{PushConstantRange};

use vulkano::buffer::{Buffer, BufferCreateInfo, BufferUsage};
use vulkano::memory::allocator::{AllocationCreateInfo, MemoryTypeFilter};

use vulkano::command_buffer::allocator::{
    StandardCommandBufferAllocator, StandardCommandBufferAllocatorCreateInfo,
};
use vulkano::command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage, CopyBufferInfo};

use vulkano::sync::{self, GpuFuture};


use vulkano::pipeline::Pipeline;
use vulkano::descriptor_set::{PersistentDescriptorSet, WriteDescriptorSet};
use vulkano::descriptor_set::allocator::StandardDescriptorSetAllocator;
use vulkano::pipeline::PipelineBindPoint;

use vulkano::pipeline::compute::ComputePipelineCreateInfo;
use vulkano::pipeline::layout::PipelineDescriptorSetLayoutCreateInfo;
use vulkano::pipeline::{ComputePipeline, PipelineShaderStageCreateInfo};

use vulkano::sync::{PipelineStages, AccessFlags};


use vulkano::descriptor_set::layout::DescriptorBindingFlags; // Required for `binding_flags`




use vulkano::shader::ShaderModule;
use std::fs::File;
use std::io::Read;
use bytemuck::{Pod, Zeroable};

use crate::shaders::render_information::RenderInformation;

use vulkano::buffer::Subbuffer;

use num_cpus;

/*
RIGHT HANDED COORDINATE SYSTEM:

+x = RIGHT
+y = UP
+z = FORWARD

*/

#[repr(C)]
#[derive(Default, Copy, Clone, Pod, Zeroable)]
pub struct Transformation {
    pub transform_matrix: [[f32; 4]; 4], // Must be an array, not `Matrix4`
}

#[repr(C)] // Ensures memory layout compatibility with Vulkan
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
struct PushConstants {
    a: f32,
    b: f32,
}

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

    
    fn load_shader(device: Arc<vulkano::device::Device>, path: &str) -> Arc<ShaderModule> {
        let mut file = File::open(path).expect("Failed to open shader file");
        let mut shader_bytes = Vec::new();
        file.read_to_end(&mut shader_bytes).unwrap();

        unsafe { ShaderModule::from_bytes(device.clone(), &shader_bytes) }
            .expect("Failed to create shader module")
}


    pub fn clip_at_edge(x1: &mut f32, y1: &mut f32, x2: &mut f32, y2: &mut f32, screen_width: i64, screen_height: i64){

        let width = screen_width as f32;
        let height =screen_height as f32;

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

    pub fn calculate_transformation(&self) -> Matrix4<f32>{

        let near = self.view.near;

        let far = self.view.far; // operates as a max render distance


        let fov = self.view.fov;  // ~70 degrees
        let depth = far-near;

        let alpha = self.view.roll;
        let beta = self.view.pitch;
        let gamma: f32 = self.view.yaw;

        let c1 = -(far+near) / depth;
        let c2 = -2.*far*near / depth; 

        // aspect ratio
        let a = self.view.aspect_ratio;

        let tan_fov = ((fov/2.) as f32).tan();

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

    fn get_mesh_vertex_indices(&self) -> Vec<usize> {


        // shud proly precompute this

        let mut idx_vec: Vec<usize> = Vec::new();    

        for mesh in &self.world.elements {
            idx_vec.push(mesh.num_vertices());
        }

        idx_vec
    }



    pub fn render(&self, render_information: &RenderInformation, pixel_buffer: &mut Vec<u32>, depth_buffer: &mut Vec<f32>, use_wireframe: bool, screen_width: i64, screen_height: i64){

        pixel_buffer.fill(0x87CEFA);



        /*
        
         
        let idx_vec = self.get_mesh_vertex_indices();

        // compute vertices
        let vertices = self.compute_vertex_screen_coordinates(render_information, screen_width, screen_height);

        let mut idx_offset = 0;

        let color = 0xFF0000;
        for (i,mesh) in self.world.elements.iter().enumerate() {

            for face in mesh.faces() {

                let p0 = vertices[(face.vertex_ids[0] + idx_offset) as usize].xy();
                let p1 = vertices[(face.vertex_ids[1] + idx_offset) as usize].xy();
                let p2 = vertices[(face.vertex_ids[2] + idx_offset) as usize].xy();


            
                let line = Line::from_vec(p0, p1, color);
                line.draw(pixel_buffer, screen_width, screen_height);

                let line = Line::from_vec(p1, p2, color);
                line.draw(pixel_buffer, screen_width, screen_height);

                let line = Line::from_vec(p2, p0, color);
                line.draw(pixel_buffer, screen_width, screen_height);

            }

            idx_offset += idx_vec[i] as i64;
        }


        return;

        */




//////////////////////


        //pixel_buffer.fill(0x000000);

        depth_buffer.fill(-INFINITY);

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
        //if use_wireframe {
            let collected_data: Vec<Vec<Vec<Vector4<f32>>>> = 
            self.world.elements.par_chunks(fragment_size).map(
                |chunk| {
    
                    let mut thread_data: Vec<Vec<Vector4<f32>>> = Vec::new();
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
                        line.draw(pixel_buffer, screen_width, screen_height);
                    }
                }
            }


            //return;
        //}



        // draw faces
        /*
        How to efficiently get the depth of the triangle face at all points on the face?

        // for now, we will just use a singular depth value, it should be good enough for simple things

        1. get normal vector
      
        
        
         */


        /*
        // face rendering
        let collected_data: Vec<Vec<Vec<Vector4<Vector2<f32>>>>> = 
        self.world.elements.par_chunks(fragment_size).map(
            |chunk| {

                let mut thread_data:  Vec<Vec<Vector4<Vector2<f32>>>> = Vec::new();
                for mesh in chunk.iter() {
                    thread_data.push(self.calculate_mesh_faces(mesh, full_transformation, screen_width, screen_height));
                }
                thread_data
            }
        ).collect();

        let mut c: u32 = 0;

        for i in collected_data.iter(){
            for j in i.iter(){
                for k in j.iter() {
                    let triangle = Triangle::from_vec4_list(k, 0xFFFFFF * (c%2));
                    triangle.draw(pixel_buffer, depth_buffer, screen_width, screen_height);
                }
            }
            c+=1;
        }
        
         */



        // draw faces (default behavior)


    }


    fn calculate_mesh_wireframe(&self, mesh: &Box<dyn Mesh>, full_transformation: Matrix4<f32>, screen_width: i64, screen_height: i64) -> Vec<Vector4<f32>>{
        let width = screen_width as f32;
        let height = screen_height as f32;


        let mut mesh_data: Vec<Vector4<f32>> = Vec::new();


        // if not in view, dont render
        if !self.view.in_view(mesh){
            return mesh_data;
        }
        
        let mut transformed_vertices: Vec<Vector4<f32>> = Vec::new();
        let mut w_vals: Vec<f32> = Vec::new();
        for point in mesh.vertices(){
            let transformed_vertex = full_transformation * point.position;
            w_vals.push(transformed_vertex[3]);
            transformed_vertices.push(transformed_vertex / transformed_vertex[3]);
            
        } 

        // render each face
        for face in mesh.faces(){

            // rendering lines
            for i in 0..3 {

                let id_a = face.vertex_ids[i] as usize;
                let id_b = face.vertex_ids[(i+1)%3] as usize;

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

    fn calculate_mesh_faces(&self, mesh: &Box<dyn Mesh>, full_transformation: Matrix4<f32>, screen_width: i64, screen_height: i64) ->  Vec<Vector4<Vector2<f32>>>{
        let width = screen_width as f32;
        let height = screen_height as f32;


        let mut mesh_data: Vec<Vector4<Vector2<f32>>> = Vec::new();


        // if not in view, dont render
        if !self.view.in_view(mesh){
            return mesh_data;
        }
        
        let mut transformed_vertices: Vec<Vector4<f32>> = Vec::new();
        let mut w_vals: Vec<f32> = Vec::new();
        for point in mesh.vertices(){
            let transformed_vertex = full_transformation * point.position;
            w_vals.push(transformed_vertex[3]);
            transformed_vertices.push(transformed_vertex / transformed_vertex[3]);
            
        } 

        // render each face
        for face in mesh.faces(){

            // the 4th vector2 represents depth, as a centroid
            let mut face_data: Vector4<Vector2<f32>> = Vector4::new(
                Vector2::new(0.,0.,),
                Vector2::new(0.,0.,),
                Vector2::new(0.,0.,),
                Vector2::new(0.,0.),
            );
            // rendering lines
            for i in 0..3 {

                let id_a = face.vertex_ids[i] as usize;

                // if w_vals[id_a] > -self.view.near || w_vals[id_b] > -self.view.near {
                //     continue;
                // }

                let a_position_prime = transformed_vertices[id_a];

                // scale back to device coordinates
                // can remove some parameter passing by scaling AFTER clipping, idc rn
                let x1_prime = width * (a_position_prime[0] +1.)/2.;
                let y1_prime = height * (1.-a_position_prime[1])/2.;

                face_data[i] =Vector2::new(x1_prime,y1_prime);

                face_data[3][0] += w_vals[id_a];
            }
            face_data[3][0] /= 3.;

            mesh_data.push(face_data);

            // render triangles
        }

        mesh_data
    }

    fn compute_vertex_screen_coordinates (&self, render_information: &RenderInformation, screen_width: i64, screen_height: i64) -> Vec<Vector4<f32>> {
        // gpu calls
        

        let push_constant_range = PushConstantRange {
    stages: ShaderStages::COMPUTE, // Use in a compute shader
    offset: 0,
    size: std::mem::size_of::<PushConstants>() as u32,
};


        /*
            Vertex Buffers
         */
        let mut raw_vertex_data: Vec<Vector4<f32>> = Vec::new();
        // put all vertex data into one list, and decompose afterward
        for mesh in &self.world.elements {
            for vertice in mesh.vertices(){
                raw_vertex_data.push(vertice.position.cast::<f32>());
            }
        }

        let buffer_size = (raw_vertex_data.len() * std::mem::size_of::<[f32; 4]>()) as u64; // Total buffer size in bytes

        let vertex_staging_buffer: Subbuffer<[[f32; 4]]> = Buffer::from_iter( 
            render_information.memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::TRANSFER_SRC,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            raw_vertex_data
            .iter() // Iterate over `Vec<[f32; 4]>`
            .map(|&v| [v[0] as f32, v[1] as f32, v[2] as f32, v[3] as f32]),

        ).expect("failed to cretae staging buffer");

        let vertex_buffer: Subbuffer<[[f32; 4]]> = Buffer::new_unsized(
            render_information.memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::STORAGE_BUFFER | BufferUsage::TRANSFER_DST | BufferUsage::TRANSFER_SRC,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
                ..Default::default()
            },
            buffer_size
        ).expect("Failed to create storage buffer!");

        let vertex_readback_buffer: Subbuffer<[[f32; 4]]> = Buffer::new_unsized(
            render_information.memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::TRANSFER_DST, // Transfer destination for readback
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::HOST_SEQUENTIAL_WRITE, // CPU-readable
                ..Default::default()
            },
            raw_vertex_data.len() as u64, // Same size as vertex_buffer
        ).expect("Failed to create readback buffer!");
        

        /*
            Transform Buffer
         */

        let mat = self.calculate_transformation().cast::<f32>();


        let transform_staging_buffer = Buffer::from_data(
            render_information.memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::TRANSFER_SRC, // UBO usage
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::HOST_SEQUENTIAL_WRITE | MemoryTypeFilter::PREFER_HOST, // Writable memory
                ..Default::default()
            },
            Transformation {
                transform_matrix: mat.into(), // Convert nalgebra Matrix4<f32> to [[f32; 4]; 4]
            },
        ).expect("Failed to create uniform buffer");


        let transform_buffer: Subbuffer<Transformation> = Buffer::new_unsized(
            render_information.memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::UNIFORM_BUFFER | BufferUsage::TRANSFER_DST,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
                ..Default::default()
            },
            std::mem::size_of::<Transformation>() as u64
        ).expect("Failed to create storage buffer!");
        


 
        let mut command_buffer_builder = AutoCommandBufferBuilder::primary(
            &render_information.command_buffer_allocator,
            render_information.queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();
    

        let work_group_counts = [1024, 1, 1];

    
    
        let shader = Renderer::load_shader(render_information.device.clone(), "src/shaders/comp.spv");
        //let shader = cs::load(device.clone()).expect("failed to create shader module");
        let cs = shader.entry_point("main").unwrap();


        let stage = PipelineShaderStageCreateInfo::new(cs);
        let layout = PipelineLayout::new(
            render_information.device.clone(),
            PipelineDescriptorSetLayoutCreateInfo::from_stages([&stage])
                .into_pipeline_layout_create_info(render_information.device.clone())
                .unwrap()
        )
        .unwrap();
    
        let compute_pipeline = ComputePipeline::new(
            render_information.device.clone(),
            None,
            ComputePipelineCreateInfo::stage_layout(stage, layout),
        )
        .expect("failed to create compute pipeline");

    
        let descriptor_set_allocator =
            StandardDescriptorSetAllocator::new(render_information.device.clone(), Default::default());
    
        let pipeline_layout = compute_pipeline.layout();
        let descriptor_set_layouts = pipeline_layout.set_layouts();
    

        let descriptor_set_layout_index = 0;
        let descriptor_set_layout = descriptor_set_layouts
            .get(descriptor_set_layout_index)
            .unwrap();

        let descriptor_set = PersistentDescriptorSet::new(
            &descriptor_set_allocator,
            descriptor_set_layout.clone(),
            [
                WriteDescriptorSet::buffer(0, vertex_buffer.clone()),
                WriteDescriptorSet::buffer(1, transform_buffer.clone()),
                //WriteDescriptorSet::buffer(1, transform_buffer.clone()),
                ], 
            [],
        )
        .unwrap();

               
        // .copy_buffer(CopyBufferInfo::buffers(transform_staging_buffer.clone(), transform_buffer.clone()))
        // .unwrap()


        let push_constants = PushConstants {
            a: screen_width as f32,
            b: screen_height as f32,  
        };
       
    
        command_buffer_builder
            .copy_buffer(CopyBufferInfo::buffers(vertex_staging_buffer.clone(), vertex_buffer.clone()))
            .unwrap()
            .copy_buffer(CopyBufferInfo::buffers(transform_staging_buffer.clone(), transform_buffer.clone()))
            .unwrap()
            .bind_pipeline_compute(compute_pipeline.clone())
            .unwrap()
            .push_constants(compute_pipeline.layout().clone(), 0, push_constants).unwrap()
            .bind_descriptor_sets(
                PipelineBindPoint::Compute,
                compute_pipeline.layout().clone(),
                descriptor_set_layout_index as u32,
                descriptor_set,
            )
            .unwrap()
            .dispatch(work_group_counts)
            .unwrap()
            .copy_buffer(CopyBufferInfo::buffers(vertex_buffer.clone(), vertex_readback_buffer.clone())) // Copy back to CPU
            .unwrap();
        
        let command_buffer = command_buffer_builder.build().unwrap();
    
        let future = sync::now(render_information.device.clone())
        .then_execute(render_information.queue.clone(), command_buffer)
        .unwrap()
        .then_signal_fence_and_flush()
        .unwrap();
    
    future.wait(None).unwrap();  // Ensures GPU completion before reading
    
    let content = vertex_readback_buffer.read().unwrap();


    let out: Vec<Vector4<f32>> = content.iter().map(|&v| Vector4::from(v).cast::<f32>()).collect();

    out
    }

}

