use crate::world::World;
use crate::view::View;
use nalgebra::Vector2;
use nalgebra::Vector3;

use rayon::prelude::*;
use vulkano::image::ImageCreateInfo;
use crate::mesh::mesh::Mesh;
use vulkano::command_buffer::ClearColorImageInfo;
use vulkano::format::ClearColorValue;


use std::time::Instant;

use crate::render::buffers::Buffers;

use crate::render::structures::{PushConstants,PushConstantsB,PushConstantsC,FrustumFaces, PushConstantsSortVertices};



use nalgebra::Matrix4;
use nalgebra::Vector4;



use vulkano::buffer::{Buffer, BufferCreateInfo, BufferUsage};
use vulkano::memory::allocator::{AllocationCreateInfo, MemoryTypeFilter};


use vulkano::command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage, CopyBufferInfo};

use vulkano::sync::{self, GpuFuture};


use vulkano::pipeline::Pipeline;
use vulkano::descriptor_set::{PersistentDescriptorSet, WriteDescriptorSet};
use vulkano::descriptor_set::allocator::StandardDescriptorSetAllocator;
use vulkano::pipeline::PipelineBindPoint;



use crate::render::render_information::RenderInformation;

use vulkano::buffer::Subbuffer;

use vulkano::command_buffer::CopyImageToBufferInfo;


use vulkano::image::ImageType;

use minifb::Window;


use std::sync::Arc;

// https://zeux.io/2020/02/27/writing-an-efficient-vulkan-renderer/?utm_source=chatgpt.com


use vulkano::image::{Image, ImageUsage, view::ImageView};
use vulkano::format::Format;

use crate::plane::Plane;
/*
RIGHT HANDED COORDINATE SYSTEM:

+x = RIGHT
+y = UP
+z = FORWARD

*/




pub struct Renderer{
    pub world: World,
    pub view: View,
    pub render_information: RenderInformation,
    pub buffers: Buffers,
    pub frame_count: u64,
    pub screen_width: usize,
    pub screen_height: usize,
    pub is_drawing_faces: bool,
}



impl Renderer {

// pass render information in

    pub fn new(world: World, view: View, screen_width: usize, screen_height: usize) -> Renderer{


        // I should organize this more

        let render_information = RenderInformation::new();


        let buffers = Buffers::new(&render_information, &world, &view,  screen_width, screen_height);

        let frame_count = 0;

        let is_drawing_faces = true;

        Renderer{
            world,
            view,
            render_information,
            buffers,
            frame_count,
            screen_width,
            screen_height,
            is_drawing_faces,
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


    pub fn render(& mut self, window: &mut Window){
        self.frame_count += 1;



        self.calculate_in_view();




        //self.sort_faces_by_tiles();
        self.compute_vertex_screen_coordinates();

        if self.is_drawing_faces {
            self.draw_faces(window);
        }else{
            self.draw_wireframe(window);
        }



    }

    fn sort_faces_by_tiles(&self){

        let chunk_width: u32 = 128;
        let num_chunks_x: u32 = (self.screen_width as f32 / chunk_width as f32).ceil() as u32;
        let num_chunks_y: u32 = (self.screen_height as f32 / chunk_width as f32).ceil() as u32;

        let buffer_length: u32 = num_chunks_x * num_chunks_y;


        let buffer_size = (buffer_length * std::mem::size_of::<u32>() as u32) as u64; // Total buffer size in bytes

    

        let tile_counts_staging_buffer: Subbuffer<[u32]> = Buffer::from_iter( 
            self.render_information.memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::TRANSFER_SRC,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            vec![0, buffer_length]

        ).expect("failed to create staging buffer");

        
        let tile_counts_buffer: Subbuffer<[u32]> = Buffer::new_unsized(
            self.render_information.memory_allocator.clone(),
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

        let tile_counts_readback_buffer: Subbuffer<[u32]> = Buffer::new_unsized(
            self.render_information.memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::TRANSFER_DST | BufferUsage::TRANSFER_SRC,  // Transfer destination for readback
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::HOST_SEQUENTIAL_WRITE, // CPU-readable
                ..Default::default()
            },
            buffer_size
        ).expect("Failed to create readback buffer!");
        

        let running_tile_counts_buffer: Subbuffer<[u32]> = Buffer::new_unsized(
            self.render_information.memory_allocator.clone(),
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


        let tile_ids_buffer: Subbuffer<[u32]> = Buffer::new_unsized(
            self.render_information.memory_allocator.clone(),
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


        let mut command_buffer_builder = AutoCommandBufferBuilder::primary(
            &self.render_information.command_buffer_allocator,
            self.render_information.queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();
    

        let descriptor_set_allocator =
            StandardDescriptorSetAllocator::new(self.render_information.device.clone(), Default::default());
    
        let pipeline_layout = self.render_information.sum_faces_by_tile_compute_pipeline.layout();
        let descriptor_set_layouts = pipeline_layout.set_layouts();

  

        let descriptor_set_layout_index = 0;
        let descriptor_set_layout = descriptor_set_layouts
            .get(descriptor_set_layout_index)
            .unwrap();

        let descriptor_set = PersistentDescriptorSet::new(
            &descriptor_set_allocator,
            descriptor_set_layout.clone(),
            [
                WriteDescriptorSet::buffer(0, self.buffers.transformed_vertex_buffer.clone()),
                WriteDescriptorSet::buffer(1, tile_counts_buffer.clone()),
                WriteDescriptorSet::buffer(2, tile_ids_buffer.clone()),
                WriteDescriptorSet::buffer(3, self.buffers.face_buffer.clone()),
                WriteDescriptorSet::buffer(4, self.buffers.running_vertice_buffer.clone()),
                ], 
            [],
        )
        .unwrap();


        let push_constants = PushConstantsSortVertices {
            a: num_chunks_x,
            b: num_chunks_y,  
            c: chunk_width,
        };
       
       let copy_operations = [
        CopyBufferInfo::buffers(tile_counts_staging_buffer.clone(), tile_counts_buffer.clone()),
        CopyBufferInfo::buffers(self.buffers.face_staging_buffer.clone(), self.buffers.face_buffer.clone()),
        CopyBufferInfo::buffers(self.buffers.running_vertice_staging_buffer.clone(), self.buffers.running_vertice_buffer.clone()),
       ];

       for copy_op in copy_operations{
        command_buffer_builder.copy_buffer(copy_op).unwrap();
       }

    
        command_buffer_builder
            .push_constants(self.render_information.sum_faces_by_tile_compute_pipeline.layout().clone(), 0, push_constants)
                .unwrap()
            .bind_pipeline_compute(self.render_information.sum_faces_by_tile_compute_pipeline.clone())
                .unwrap()
            .bind_descriptor_sets(
                PipelineBindPoint::Compute,
                self.render_information.sum_faces_by_tile_compute_pipeline.layout().clone(),
                descriptor_set_layout_index as u32,
                descriptor_set,
            )
                .unwrap()

            .dispatch(self.render_information.work_group_counts)
                .unwrap();

        let command_buffer = command_buffer_builder.build().unwrap();
    
        let future = sync::now(self.render_information.device.clone())
        .then_execute(self.render_information.queue.clone(), command_buffer)
            .unwrap()
        .then_signal_fence_and_flush()
            .unwrap();

    
        future.wait(None).unwrap();  // Ensures GPU completion before reading


    }

    fn compute_vertex_screen_coordinates (&self){ //} -> Vec<Vector4<f32>> 



        let mut write_lock = self.buffers.transform_staging_buffers[(self.frame_count%2) as usize].write().unwrap();
        write_lock.transform_matrix = self.calculate_transformation().into();

        let mut command_buffer_builder: AutoCommandBufferBuilder<vulkano::command_buffer::PrimaryAutoCommandBuffer> = AutoCommandBufferBuilder::primary(
            &self.render_information.command_buffer_allocator,
            self.render_information.queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();
    


        let descriptor_set_allocator =
            StandardDescriptorSetAllocator::new(self.render_information.device.clone(), Default::default());
    
        let pipeline_layout = self.render_information.vertex_compute_pipeline.layout();
        let descriptor_set_layouts = pipeline_layout.set_layouts();
    

        let descriptor_set_layout_index = 0;
        let descriptor_set_layout = descriptor_set_layouts
            .get(descriptor_set_layout_index)
            .unwrap();



        let descriptor_set = PersistentDescriptorSet::new(
            &descriptor_set_allocator,
            descriptor_set_layout.clone(),
            [
                WriteDescriptorSet::buffer(0, self.buffers.vertex_buffer.clone()),
                WriteDescriptorSet::buffer(1, self.buffers.transformed_vertex_buffer.clone()),
                WriteDescriptorSet::buffer(2, self.buffers.transform_buffer.clone()),
                //WriteDescriptorSet::buffer(1, transform_buffer.clone()),
                ], 
            [],
        )
        .unwrap();

               
        // .copy_buffer(CopyBufferInfo::buffers(transform_staging_buffer.clone(), transform_buffer.clone()))
        // .unwrap()


        let push_constants = PushConstants {
            a: self.screen_width as f32,
            b: self.screen_height as f32,  
        };

       
    
        command_buffer_builder
            .copy_buffer(CopyBufferInfo::buffers(self.buffers.vertex_staging_buffer.clone(), self.buffers.vertex_buffer.clone()))
                .unwrap()
            .copy_buffer(CopyBufferInfo::buffers(self.buffers.transform_staging_buffers[((self.frame_count+1) %2) as usize].clone(), self.buffers.transform_buffer.clone()))
                .unwrap()
            .bind_pipeline_compute(self.render_information.vertex_compute_pipeline.clone())
                .unwrap()
            .push_constants(self.render_information.vertex_compute_pipeline.layout().clone(), 0, push_constants).unwrap()
            .bind_descriptor_sets(
                PipelineBindPoint::Compute,
                self.render_information.vertex_compute_pipeline.layout().clone(),
                descriptor_set_layout_index as u32,
                descriptor_set,
            )
            .unwrap()
            .dispatch(self.render_information.work_group_counts)
            .unwrap().copy_buffer(CopyBufferInfo::buffers(self.buffers.vertex_buffer.clone(), self.buffers.vertex_readback_buffer.clone())) // Copy back to CPU
                .unwrap();


        let command_buffer = command_buffer_builder.build().unwrap();

        let future = sync::now(self.render_information.device.clone())
        .then_execute(self.render_information.queue.clone(), command_buffer)
        .unwrap()
        .then_signal_fence_and_flush()
        .unwrap();
    
    future.wait(None).unwrap();  // Ensures GPU completion before reading


    // let content = self.buffers.vertex_readback_buffer.read().unwrap();
    // println!("here!:");
    // println!("{} {} {}", self.view.x,self.view.y,self.view.z);
    // for v in content.iter(){
    //     println!("{:?}", v);
    // }


    
    }


    fn draw_wireframe(&self, window: &mut Window) {


        let image = Image::new(
            self.render_information.memory_allocator.clone(),
            ImageCreateInfo {
                image_type: ImageType::Dim2d,
                format: Format::R8G8B8A8_UNORM,
                extent: [self.screen_width as u32,self.screen_height as u32, 1],
                usage: ImageUsage::STORAGE | ImageUsage::TRANSFER_SRC | ImageUsage::TRANSFER_DST,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
                ..Default::default()
            },
        )
        .unwrap();


        let image_view = ImageView::new_default(image.clone()).unwrap();

        let image_size = self.screen_width * self.screen_height * 4; // RGBA, 4 bytes per pixel
        let output_img_buf: Subbuffer<[u32]> = Buffer::new_slice(
            self.render_information.memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::TRANSFER_DST | BufferUsage::TRANSFER_SRC, // Can be read and transferred
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::HOST_RANDOM_ACCESS,
                ..Default::default()
            },
            image_size as u64, // Corrected buffer size
        ).expect("Failed to create readback buffer!");


        let mut command_buffer_builder = AutoCommandBufferBuilder::primary(
            &self.render_information.command_buffer_allocator,
            self.render_information.queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();
    

        let descriptor_set_allocator =
            StandardDescriptorSetAllocator::new(self.render_information.device.clone(), Default::default());
    
        let pipeline_layout = self.render_information.line_draw_compute_pipeline.layout();
        let descriptor_set_layouts = pipeline_layout.set_layouts();

  

        let descriptor_set_layout_index = 0;
        let descriptor_set_layout = descriptor_set_layouts
            .get(descriptor_set_layout_index)
            .unwrap();

        let descriptor_set = PersistentDescriptorSet::new(
            &descriptor_set_allocator,
            descriptor_set_layout.clone(),
            [
                WriteDescriptorSet::buffer(0, self.buffers.face_buffer.clone()),
                WriteDescriptorSet::buffer(1, self.buffers.running_vertice_buffer.clone()),
                WriteDescriptorSet::buffer(2, self.buffers.in_view_buffer.clone()),
                WriteDescriptorSet::buffer(3, self.buffers.transformed_vertex_buffer.clone()),
                WriteDescriptorSet::image_view(4, image_view.clone()),

                ], 
            [],
        )
        .unwrap();

        let dist_scale: f32 = self.view.near / self.view.dist_from_window; // check calculations 

        let push_constants = PushConstantsB {
            a: self.screen_width as f32,
            b: self.screen_height as f32,  
            c: dist_scale as f32,
        };
       
       let copy_operations = [
        CopyBufferInfo::buffers(self.buffers.face_staging_buffer.clone(), self.buffers.face_buffer.clone()),
        CopyBufferInfo::buffers(self.buffers.running_vertice_staging_buffer.clone(), self.buffers.running_vertice_buffer.clone()),
       ];

       for copy_op in copy_operations{
        command_buffer_builder.copy_buffer(copy_op).unwrap();
       }

    
        command_buffer_builder
            .push_constants(self.render_information.line_draw_compute_pipeline.layout().clone(), 0, push_constants)
                .unwrap()
            .bind_pipeline_compute(self.render_information.line_draw_compute_pipeline.clone())
                .unwrap()
            .bind_descriptor_sets(
                PipelineBindPoint::Compute,
                self.render_information.line_draw_compute_pipeline.layout().clone(),
                descriptor_set_layout_index as u32,
                descriptor_set,
            )
                .unwrap()
            .clear_color_image(ClearColorImageInfo {
                clear_value: ClearColorValue::Float([0.98, 0.94, 0.82, 1.0]),
                ..ClearColorImageInfo::image(image.clone())
            })
                .unwrap()
            .dispatch(self.render_information.work_group_counts)
                .unwrap()
            .copy_image_to_buffer(CopyImageToBufferInfo::image_buffer(
                image.clone(),
                output_img_buf.clone(),
            ))
            .unwrap();

        let command_buffer = command_buffer_builder.build().unwrap();
    
        let future = sync::now(self.render_information.device.clone())
        .then_execute(self.render_information.queue.clone(), command_buffer)
            .unwrap()
        .then_signal_fence_and_flush()
            .unwrap();

    
        future.wait(None).unwrap();  // Ensures GPU completion before reading


        let content = output_img_buf.read().unwrap();

        window.update_with_buffer(&content, self.screen_width, self.screen_height).unwrap();


    }

    fn calculate_in_view(&self){ //} -> Vec<u32>{

        // let now = Instant::now();
        // let elapsed = now.elapsed().as_nanos();

        // println!("Time elapsed in compute vertex_ {}", elapsed);

        

        let mut frustum_content = self.buffers.frustum_faces_staging_buffers[(self.frame_count%2) as usize].write().unwrap();

        let frustum_faces = &self.view.frustum_faces;


        *frustum_content = FrustumFaces {
            faces: frustum_faces.iter()
                .map(|face: &Plane| [face.normal[0], face.normal[1], face.normal[2], face.distance])
                .collect::<Vec<[f32; 4]>>()
                .try_into()
                .expect("Expected exactly 6 faces"),
        };

        // let now = Instant::now();

        
        let mut command_buffer_builder = AutoCommandBufferBuilder::primary(
            &self.render_information.command_buffer_allocator,
            self.render_information.queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();
    

        let descriptor_set_allocator =
            StandardDescriptorSetAllocator::new(self.render_information.device.clone(), Default::default());
    
        let pipeline_layout = self.render_information.in_view_compute_pipeline.layout();
        let descriptor_set_layouts = pipeline_layout.set_layouts();

        let descriptor_set_layout_index = 0;
        let descriptor_set_layout = descriptor_set_layouts
            .get(descriptor_set_layout_index)
            .unwrap();

        let descriptor_set = PersistentDescriptorSet::new(
            &descriptor_set_allocator,
            descriptor_set_layout.clone(),
            [
                WriteDescriptorSet::buffer(0, self.buffers.bounding_box_buffer.clone()),
                WriteDescriptorSet::buffer(1,self.buffers.frustum_faces_buffer.clone()),
                WriteDescriptorSet::buffer(2, self.buffers.in_view_buffer.clone()),
                ], 
            [],
        )
        .unwrap();     


        // let elapsed = now.elapsed().as_nanos();

        // println!("Time elapsed in compute vertex_ {}", elapsed);



        command_buffer_builder
            .copy_buffer(CopyBufferInfo::buffers(self.buffers.bounding_box_staging_buffer.clone(), self.buffers.bounding_box_buffer.clone()))
                .unwrap()
            .copy_buffer(CopyBufferInfo::buffers(self.buffers.frustum_faces_staging_buffers[((self.frame_count+1)%2) as usize].clone(), self.buffers.frustum_faces_buffer.clone()))
                .unwrap()
            .bind_pipeline_compute(self.render_information.in_view_compute_pipeline.clone())
            .   unwrap()
            .bind_descriptor_sets(
                PipelineBindPoint::Compute,
                self.render_information.in_view_compute_pipeline.layout().clone(),
                descriptor_set_layout_index as u32,
                descriptor_set,
            )
                .unwrap()
            .dispatch(self.render_information.work_group_counts)
                .unwrap();


        let command_buffer = command_buffer_builder.build().unwrap();


    
        let future = sync::now(self.render_information.device.clone())
        .then_execute(self.render_information.queue.clone(), command_buffer)
        .unwrap()
        .then_signal_fence_and_flush()
        .unwrap();
    
    future.wait(None).unwrap();  // Ensures GPU completion before reading


        



    }


    fn draw_faces(&self, window: &mut Window){

        let mut command_buffer_builder = AutoCommandBufferBuilder::primary(
            &self.render_information.command_buffer_allocator,
            self.render_information.queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();


        let descriptor_set_allocator =
            StandardDescriptorSetAllocator::new(self.render_information.device.clone(), Default::default());
    
        let pipeline_layout = self.render_information.triangle_draw_compute_pipeline.layout();
        let descriptor_set_layouts = pipeline_layout.set_layouts();

  

        let descriptor_set_layout_index = 0;
        let descriptor_set_layout = descriptor_set_layouts
            .get(descriptor_set_layout_index)
            .unwrap();

        let descriptor_set = PersistentDescriptorSet::new(
            &descriptor_set_allocator,
            descriptor_set_layout.clone(),
            [
                WriteDescriptorSet::buffer(0, self.buffers.face_buffer.clone()),
                WriteDescriptorSet::buffer(1, self.buffers.running_vertice_buffer.clone()),
                WriteDescriptorSet::buffer(2, self.buffers.in_view_buffer.clone()),
                WriteDescriptorSet::buffer(3, self.buffers.transformed_vertex_buffer.clone()),
                WriteDescriptorSet::buffer(4, self.buffers.depth_buffer.clone()),
                WriteDescriptorSet::buffer(5, self.buffers.color_buffer.clone()),
                WriteDescriptorSet::buffer(6, self.buffers.face_normal_buffer.clone()),
                WriteDescriptorSet::buffer(7, self.buffers.vertex_buffer.clone()),
                WriteDescriptorSet::buffer(8, self.buffers.point_light_buffer.clone()),
                WriteDescriptorSet::image_view(9, self.buffers.pixel_image_view.clone()),
                ], 
            [],
        )
        .unwrap();

        let push_constants = PushConstantsC {
            screen_width: self.screen_width as f32,
            screen_height: self.screen_height as f32,
            heading_x: self.view.direction[0],
            heading_y: self.view.direction[1],
            heading_z: self.view.direction[2],
            x: self.view.x,
            y: self.view.y,
            z: self.view.z,
        };

       let copy_operations = [
        CopyBufferInfo::buffers(self.buffers.face_staging_buffer.clone(), self.buffers.face_buffer.clone()),
        CopyBufferInfo::buffers(self.buffers.running_vertice_staging_buffer.clone(), self.buffers.running_vertice_buffer.clone()),
        CopyBufferInfo::buffers(self.buffers.color_staging_buffer.clone(), self.buffers.color_buffer.clone()),
        CopyBufferInfo::buffers(self.buffers.depth_staging_buffer.clone(), self.buffers.depth_buffer.clone()),
        CopyBufferInfo::buffers(self.buffers.face_normal_staging_buffer.clone(), self.buffers.face_normal_buffer.clone()),
        CopyBufferInfo::buffers(self.buffers.point_light_staging_buffer.clone(), self.buffers.point_light_buffer.clone()),
       ];

       for copy_op in copy_operations{
        command_buffer_builder.copy_buffer(copy_op).unwrap();
       }

    
        command_buffer_builder
            .bind_pipeline_compute(self.render_information.triangle_draw_compute_pipeline.clone())
                .unwrap()
            .push_constants(self.render_information.triangle_draw_compute_pipeline.layout().clone(), 0, push_constants)
                .unwrap()
            .bind_descriptor_sets(
                PipelineBindPoint::Compute,
                self.render_information.triangle_draw_compute_pipeline.layout().clone(),
                descriptor_set_layout_index as u32,
                descriptor_set,
            )
                .unwrap()
            .clear_color_image(ClearColorImageInfo {
                clear_value: ClearColorValue::Float([0.98, 0.94, 0.82, 1.0]),
                ..ClearColorImageInfo::image(self.buffers.pixel_image.clone())
            })
                .unwrap()
            .dispatch(self.render_information.work_group_counts)
                .unwrap()
            .copy_image_to_buffer(CopyImageToBufferInfo::image_buffer(
                self.buffers.pixel_image.clone(),
                self.buffers.output_img_buf.clone(),
            ))
            .unwrap();
        let command_buffer = command_buffer_builder.build().unwrap();
    



        let future = sync::now(self.render_information.device.clone())
        .then_execute(self.render_information.queue.clone(), command_buffer)
            .unwrap()
        .then_signal_fence_and_flush()
            .unwrap();

    
        future.wait(None).unwrap();  // Ensures GPU completion before reading



        let content = self.buffers.output_img_buf.read().unwrap();

        window.update_with_buffer(&content, self.screen_width, self.screen_height).unwrap();

    }
}