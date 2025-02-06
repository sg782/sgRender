use std::f32::INFINITY;

use crate::world::World;
use crate::view::View;
use nalgebra::Vector2;
use nalgebra::Vector3;

use rayon::prelude::*;
use vulkano::image::ImageCreateInfo;
use crate::mesh::mesh::Mesh;
use vulkano::command_buffer::ClearColorImageInfo;
use vulkano::format::ClearColorValue;

use crate::primitives::line::Line;

use crate::primitives::triangle::Triangle;


use nalgebra::Matrix4;
use nalgebra::Vector4;


use vulkano::VulkanLibrary;
use vulkano::instance::{Instance, InstanceCreateFlags, InstanceCreateInfo};
use vulkano::device::QueueFlags;

use vulkano::device::{Device, DeviceCreateInfo, QueueCreateInfo};


use vulkano::buffer::{Buffer, BufferCreateInfo, BufferUsage};
use vulkano::memory::allocator::{AllocationCreateInfo, MemoryTypeFilter};


use vulkano::command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage, CopyBufferInfo};

use vulkano::sync::{self, GpuFuture};


use vulkano::pipeline::Pipeline;
use vulkano::descriptor_set::{PersistentDescriptorSet, WriteDescriptorSet};
use vulkano::descriptor_set::allocator::StandardDescriptorSetAllocator;
use vulkano::pipeline::PipelineBindPoint;

use bytemuck::{Pod, Zeroable};

use crate::shaders::render_information::RenderInformation;

use vulkano::buffer::Subbuffer;

use num_cpus;
use vulkano::command_buffer::CopyImageToBufferInfo;


use vulkano::image::ImageType;

use minifb::Window;


use vulkano::image::{Image, ImageUsage, view::ImageView};
use vulkano::format::Format;

use crate::plane::Plane;
/*
RIGHT HANDED COORDINATE SYSTEM:

+x = RIGHT
+y = UP
+z = FORWARD

*/

#[repr(C)]
#[derive(Default, Copy, Clone, Pod, Zeroable)]
pub struct Transformation {
    pub transform_matrix: [[f32; 4]; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct BoundingPoint {
    min: [f32 ; 3],
    _pad1: f32,
    max: [f32 ; 3],
    _pad2: f32,

}

#[repr(C)] 
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
pub struct FrustumFaces {
    faces: [[ f32; 4];6], // normal: Vec3<f32>, and Distance: f32
}

#[repr(C)] 
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
struct PushConstants {
    a: f32,
    b: f32,
}

#[repr(C)] 
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
struct PushConstantsB {
    a: f32,
    b: f32,
    c: f32,
}

#[repr(C)] 
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
struct PushConstantsC {
    screen_width: f32,
    screen_height: f32,
    roll: f32,
    pitch: f32, 
    yaw: f32,
    x: f32, 
    y: f32,
    z: f32,
}

pub struct Buffers { 

    // vertices
    pub vertex_staging_buffer: Subbuffer<[[f32; 4]]>,
    pub vertex_buffer: Subbuffer<[[f32; 4]]>,
    pub vertex_readback_buffer: Subbuffer<[[f32; 4]]>,

    // rotation transform
    pub transform_staging_buffers: Vector2<Subbuffer<Transformation>>,
    pub transform_buffer: Subbuffer<Transformation>,

    //faces_buffer
    pub face_staging_buffer: Subbuffer<[[u32; 4]]>,
    pub face_buffer: Subbuffer<[[u32; 4]]>,

    // running vertices
    pub running_vertice_staging_buffer: Subbuffer<[u32]>,
    pub running_vertice_buffer: Subbuffer<[u32]>,

    pub bounding_box_staging_buffer: Subbuffer<[BoundingPoint]>,
    pub bounding_box_buffer: Subbuffer<[BoundingPoint]>,
}

pub struct Renderer{
    pub world: World,
    pub view: View,
    pub render_information: RenderInformation,
    pub buffers: Buffers,
    pub frame_count: u64,
    pub screen_width: usize,
    pub screen_height: usize,
}

impl Renderer {


// pass render information in

    pub fn new(world: World, view: View, screen_width: usize, screen_height: usize) -> Renderer{


        let library = VulkanLibrary::new().expect("no local Vulkan library/DLL");
        let instance = Instance::new(
            library,
            InstanceCreateInfo {
                flags: InstanceCreateFlags::ENUMERATE_PORTABILITY,
                ..Default::default()
            },
        )
        .expect("failed to create instance");
    
    
        let physical_device = instance
            .enumerate_physical_devices()
            .expect("could not enumerate devices")
            .next()
            .expect("no devices available");
    
        let queue_family_index = physical_device
        .queue_family_properties()
        .iter()
        .enumerate()
        .position(|(_queue_family_index, queue_family_properties)| {
            queue_family_properties.queue_flags.contains(QueueFlags::GRAPHICS)
        })
        .expect("couldn't find a graphical queue family") as u32;
    
        let (device, mut queues) = Device::new(
            physical_device,
            DeviceCreateInfo {
                // here we pass the desired queue family to use by index
                queue_create_infos: vec![QueueCreateInfo {
                    queue_family_index,
                    ..Default::default()
                }],
                ..Default::default()
            },
        )
        .expect("failed to create device");
    
        let queue = queues.next().unwrap();

        
        let mut raw_vertex_data: Vec<Vector4<f32>> = Vec::new();
        // put all vertex data into one list, and decompose afterward
        for mesh in &world.elements {
            for vertice in mesh.vertices(){

                raw_vertex_data.push(vertice.position.cast::<f32>());
            }
        }
    
        let render_information = RenderInformation::new(device.clone(), queue.clone(), &raw_vertex_data);



        // create buffers
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
                usage: BufferUsage::TRANSFER_DST | BufferUsage::TRANSFER_SRC,  // Transfer destination for readback
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::HOST_SEQUENTIAL_WRITE, // CPU-readable
                ..Default::default()
            },
            raw_vertex_data.len() as u64, // Same size as vertex_buffer
        ).expect("Failed to create readback buffer!");
        


        //let mat = self.calculate_transformation().cast::<f32>();

        let mat: Matrix4<f32> = Matrix4::zeros();


        let transform_staging_buffer_1 = Buffer::from_data(
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

        
        let transform_staging_buffer_2 = Buffer::from_data(
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
        

        let transform_staging_buffers = Vector2::new(
            transform_staging_buffer_1,
            transform_staging_buffer_2,
        );



        let mut face_data: Vec<Vector4<usize>> = Vec::new();
        for (i,mesh) in world.elements.iter().enumerate() {
            for face in mesh.faces(){
                face_data.push(Vector4::new(
                    face.vertex_ids[0] as usize,
                    face.vertex_ids[1] as usize, 
                    face.vertex_ids[2] as usize, 
                    i
                ));
            }
        }

        let buffer_size = (face_data.len() * std::mem::size_of::<[usize; 4]>()) as u64; // Total buffer size in bytes

        let face_staging_buffer: Subbuffer<[[u32; 4]]> = Buffer::from_iter( 
            render_information.memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::TRANSFER_SRC,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            face_data
            .iter() // Iterate over `Vec<[f32; 4]>`
            .map(|&v| [v[0] as u32, v[1] as u32, v[2] as u32, v[3] as u32]),

        ).expect("failed to cretae staging buffer");

        
        let face_buffer: Subbuffer<[[u32; 4]]> = Buffer::new_unsized(
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


        let vertice_indices = &world.idx_vec_running;

        let running_vertice_staging_buffer: Subbuffer<[u32]> = Buffer::from_iter( 
            render_information.memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::TRANSFER_SRC,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            vertice_indices.iter().cloned(),

        ).expect("failed to cretae staging buffer");

        
        let running_vertice_buffer: Subbuffer<[u32]> = Buffer::new_unsized(
            render_information.memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::STORAGE_BUFFER | BufferUsage::TRANSFER_DST | BufferUsage::TRANSFER_SRC,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
                ..Default::default()
            },
            vertice_indices.len() as u64,
        ).expect("Failed to create storage buffer!");


        let mut bounding_boxes: Vec<Vector2<Vector3<f32>>> = Vec::new();
        for mesh in &world.elements {
            bounding_boxes.push(*mesh.bounding_box());
        }

        let bounding_box_staging_buffer: Subbuffer<[BoundingPoint]> = Buffer::from_iter(
            render_information.memory_allocator.clone(), 
            BufferCreateInfo {
                usage: BufferUsage::TRANSFER_SRC,
                ..Default::default()
            }, 
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            }, 
            bounding_boxes.iter().map(
                |bb| BoundingPoint {
                min: [bb[0][0], bb[0][1], bb[0][2]],
                _pad1: 0.,
                max: [bb[1][0], bb[1][1], bb[1][2]],
                _pad2: 0.,
                }
            ),
        ).expect("Failed to create buffer");

        let bounding_box_buffer: Subbuffer<[BoundingPoint]> = Buffer::new_unsized(
            render_information.memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::STORAGE_BUFFER | BufferUsage::TRANSFER_DST,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
                ..Default::default()
            },
            std::mem::size_of::<BoundingPoint>() as u64 * bounding_boxes.len() as u64
        ).expect("Failed to create storage buffer!");

        // buffer encapsulation struct
        let buffers = Buffers {
            vertex_staging_buffer,
            vertex_buffer,
            vertex_readback_buffer,
            transform_buffer,
            transform_staging_buffers,

            face_staging_buffer,
            face_buffer,

            running_vertice_staging_buffer,
            running_vertice_buffer,

            bounding_box_staging_buffer,
            bounding_box_buffer,



        };



        let frame_count = 0;

        Renderer{
            world,
            view,
            render_information,
            buffers,
            frame_count,
            screen_width,
            screen_height,
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


    pub fn render(& mut self, pixel_buffer: &mut Vec<u32>, window: &mut Window,depth_buffer: &mut Vec<f32>, use_wireframe: bool, screen_width: i64, screen_height: i64){

        self.frame_count += 1;

        self.compute_vertex_screen_coordinates(screen_width, screen_height);

        //self.draw_wireframe(window, screen_width, screen_height);

        self.draw_faces(window);

        return;



        let num_cores = num_cpus::get();
        let fragment_size = (&self.world.elements.len() / num_cores) + 1;




        // draw faces
        /*
        How to efficiently get the depth of the triangle face at all points on the face?

        // for now, we will just use a singular depth value, it should be good enough for simple things

        1. get normal vector
      
        
        
         */


        let render_faces = false;

        if(render_faces){
            let collected_data: Vec<Vec<Vec<Vector4<Vector2<f32>>>>> = 
            self.world.elements.par_chunks(fragment_size).map(
                |chunk| {
    
                    let mut thread_data:  Vec<Vec<Vector4<Vector2<f32>>>> = Vec::new();
                    for mesh in chunk.iter() {
                        thread_data.push(self.calculate_mesh_faces(mesh, self.calculate_transformation(), screen_width, screen_height));
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
            
        }
        // face rendering

        // draw faces (default behavior)


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

    fn compute_vertex_screen_coordinates (&self, screen_width: i64, screen_height: i64) -> Vec<Vector4<f32>> {


        let mut write_lock = self.buffers.transform_staging_buffers[(self.frame_count%2) as usize].write().unwrap();
        write_lock.transform_matrix = self.calculate_transformation().into();

        let mut command_buffer_builder = AutoCommandBufferBuilder::primary(
            &self.render_information.command_buffer_allocator,
            self.render_information.queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();
    

        let work_group_counts = [512, 1, 1];


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
                WriteDescriptorSet::buffer(1, self.buffers.transform_buffer.clone()),
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
            .dispatch(work_group_counts)
            .unwrap()
            .copy_buffer(CopyBufferInfo::buffers(self.buffers.vertex_buffer.clone(), self.buffers.vertex_readback_buffer.clone())) // Copy back to CPU
            .unwrap();
        
        let command_buffer = command_buffer_builder.build().unwrap();
    
        let future = sync::now(self.render_information.device.clone())
        .then_execute(self.render_information.queue.clone(), command_buffer)
        .unwrap()
        .then_signal_fence_and_flush()
        .unwrap();
    
    future.wait(None).unwrap();  // Ensures GPU completion before reading
    
    let content = self.buffers.vertex_readback_buffer.read().unwrap();


    let mut out: Vec<Vector4<f32>> = content.iter().map(|&v| Vector4::from(v)).collect();

    out
    }


    fn draw_wireframe(&self, window: &mut Window, screen_width: i64, screen_height: i64) {

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

        // Wrap in ImageView	Use ImageView::new_default(image)	Allows shaders to access RawImage
        // Bind to Descriptor Set	Use WriteDescriptorSet::image_view()	Makes image available to the compute shader
        // Use imageStore() in GLSL	Write pixels directly to image2D	Avoids slow buffer copies
        // Copy to Swapchain	Use copy_image() after compute dispatch	Displays rendered pixels


        // println!("-=-=-=-=-=-=-=-");
        // println!("A: {:?}", in_vec);


        
        let in_vec = self.calculate_in_view();


        // println!("B: {:?}", in_veca);

        //println!("inVec: {:?}", in_vec);


        let buffer_size = (in_vec.len() * std::mem::size_of::<u32>()) as u64; // Total buffer size in bytes


        let in_view_staging_buffer: Subbuffer<[u32]> = Buffer::from_iter( 
            self.render_information.memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::TRANSFER_SRC,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            in_vec.iter().cloned(),

        ).expect("failed to cretae staging buffer");

        
        let in_view_buffer: Subbuffer<[u32]> = Buffer::new_unsized(
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
    

        

        let work_group_counts = [512,1,1];


        


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
                WriteDescriptorSet::buffer(2, in_view_buffer.clone()),
                WriteDescriptorSet::buffer(3, self.buffers.vertex_buffer.clone()),
                WriteDescriptorSet::image_view(4, image_view.clone()),

                ], 
            [],
        )
        .unwrap();

               
        // .copy_buffer(CopyBufferInfo::buffers(transform_staging_buffer.clone(), transform_buffer.clone()))
        // .unwrap()

        
        let dist_scale: f32 = self.view.near / self.view.dist_from_window; // check calculations 



        let push_constants = PushConstantsB {
            a: screen_width as f32,
            b: screen_height as f32,  
            c: dist_scale as f32,
        };
       
       let copy_operations = [
        CopyBufferInfo::buffers(self.buffers.face_staging_buffer.clone(), self.buffers.face_buffer.clone()),
        CopyBufferInfo::buffers(self.buffers.running_vertice_staging_buffer.clone(), self.buffers.running_vertice_buffer.clone()),
        CopyBufferInfo::buffers(in_view_staging_buffer.clone(), in_view_buffer.clone()),
        CopyBufferInfo::buffers(self.buffers.vertex_readback_buffer.clone(), self.buffers.vertex_buffer.clone()),
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
                clear_value: ClearColorValue::Float([0.0, 0.0, 0.0, 1.0]),
                ..ClearColorImageInfo::image(image.clone())
            })
                .unwrap()
            .dispatch(work_group_counts)
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

        let start = std::time::Instant::now();

        let mut content = output_img_buf.read().unwrap();

        window.update_with_buffer(&content, self.screen_width, self.screen_height).unwrap();


    }

    fn calculate_in_view(&self) -> Vec<u32>{

        let frustum_faces = &self.view.frustum_faces;

        let frustum_faces_staging_buffer:Subbuffer<FrustumFaces> = Buffer::from_data(
            self.render_information.memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::TRANSFER_SRC, // UBO usage
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::HOST_SEQUENTIAL_WRITE | MemoryTypeFilter::PREFER_HOST, // Writable memory
                ..Default::default()
            },
            FrustumFaces {
                faces: frustum_faces.iter().map(|face:&Plane| [face.normal[0], face.normal[1], face.normal[2], face.distance])
                    .collect::<Vec<[f32; 4]>>()
                    .try_into()
                    .expect("Expected exactly 6 faces"),
            },
        ).expect("Failed to create uniform buffer");


        let frustum_faces_buffer: Subbuffer<FrustumFaces> = Buffer::new_unsized(
            self.render_information.memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::UNIFORM_BUFFER | BufferUsage::TRANSFER_DST,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
                ..Default::default()
            },
            std::mem::size_of::<FrustumFaces>() as u64
        ).expect("Failed to create storage buffer!");



        
        let in_view_buffer: Subbuffer<[u32]> = Buffer::new_unsized(
            self.render_information.memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::STORAGE_BUFFER | BufferUsage::TRANSFER_DST | BufferUsage::TRANSFER_SRC,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
                ..Default::default()
            },
            self.world.elements.len() as u64,
        ).expect("Failed to create storage buffer!");

        let in_view_readback_buffer: Subbuffer<[u32]> = Buffer::new_unsized(
            self.render_information.memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::TRANSFER_DST | BufferUsage::TRANSFER_SRC,  // Transfer destination for readback
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::HOST_SEQUENTIAL_WRITE, // CPU-readable
                ..Default::default()
            },
            self.world.elements.len() as u64,
        ).expect("Failed to create readback buffer!");



    
        
        let mut command_buffer_builder = AutoCommandBufferBuilder::primary(
            &self.render_information.command_buffer_allocator,
            self.render_information.queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();
    
        let work_group_counts = [512, 1, 1];

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
                WriteDescriptorSet::buffer(1,frustum_faces_buffer.clone()),
                WriteDescriptorSet::buffer(2, in_view_buffer.clone()),
                ], 
            [],
        )
        .unwrap();     


     
               
        // .copy_buffer(CopyBufferInfo::buffers(transform_staging_buffer.clone(), transform_buffer.clone()))
        // .unwrap()


    
        command_buffer_builder
            .copy_buffer(CopyBufferInfo::buffers(self.buffers.bounding_box_staging_buffer.clone(), self.buffers.bounding_box_buffer.clone()))
                .unwrap()
            .copy_buffer(CopyBufferInfo::buffers(frustum_faces_staging_buffer.clone(), frustum_faces_buffer.clone()))
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
            .dispatch(work_group_counts)
                .unwrap()
            .copy_buffer(CopyBufferInfo::buffers(in_view_buffer.clone(), in_view_readback_buffer.clone()))
                .unwrap();

        let command_buffer = command_buffer_builder.build().unwrap();
    
        let future = sync::now(self.render_information.device.clone())
        .then_execute(self.render_information.queue.clone(), command_buffer)
        .unwrap()
        .then_signal_fence_and_flush()
        .unwrap();
    
    future.wait(None).unwrap();  // Ensures GPU completion before reading
    
    let content = in_view_readback_buffer.read().unwrap();


    let mut out: Vec<u32> = Vec::new();

    for val in content.iter() {
        out.push(*val);
    }

    

    out
        
    }

    fn draw_faces(&self, window: &mut Window){

        let pixel_image = Image::new(
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
        
        let pixel_image_view = ImageView::new_default(pixel_image.clone()).unwrap();

        let pixel_image_size = self.screen_width * self.screen_height * 4; // RGBA, 4 bytes per pixel
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
            pixel_image_size as u64, // Corrected buffer size
        ).expect("Failed to create readback buffer!");


        let in_vec = self.calculate_in_view();


        // println!("B: {:?}", in_veca);

        //println!("inVec: {:?}", in_vec);


        let buffer_size = (in_vec.len() * std::mem::size_of::<u32>()) as u64; // Total buffer size in bytes


        let in_view_staging_buffer: Subbuffer<[u32]> = Buffer::from_iter( 
            self.render_information.memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::TRANSFER_SRC,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            in_vec.iter().cloned(),

        ).expect("failed to cretae staging buffer");

        
        let in_view_buffer: Subbuffer<[u32]> = Buffer::new_unsized(
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



        let mut depth_vec: Vec<f32> = vec![-INFINITY; (self.screen_width * self.screen_height) as usize]; 
        let buffer_size = (depth_vec.len() * std::mem::size_of::<u32>()) as u64; // Total buffer size in bytes



        let depth_staging_buffer: Subbuffer<[f32]> = Buffer::from_iter( 
            self.render_information.memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::TRANSFER_SRC,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            depth_vec.iter().cloned(),

        ).expect("failed to cretae depth buffer");

        
        let depth_buffer: Subbuffer<[u32]> = Buffer::new_unsized(
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





        // println!("B: {:?}", in_veca);

        //println!("inVec: {:?}", in_vec);



        let mut command_buffer_builder = AutoCommandBufferBuilder::primary(
            &self.render_information.command_buffer_allocator,
            self.render_information.queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();
    
        let work_group_counts = [512,1,1];


    

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
                WriteDescriptorSet::buffer(2, in_view_buffer.clone()),
                WriteDescriptorSet::buffer(3, self.buffers.vertex_buffer.clone()),
                WriteDescriptorSet::buffer(4, depth_buffer.clone()),
                WriteDescriptorSet::image_view(5, pixel_image_view.clone()),
                ], 
            [],
        )
        .unwrap();

               
        // .copy_buffer(CopyBufferInfo::buffers(transform_staging_buffer.clone(), transform_buffer.clone()))
        // .unwrap()


        let push_constants = PushConstantsC {
            screen_width: self.screen_width as f32,
            screen_height: self.screen_height as f32,
            roll: self.view.roll,
            pitch: self.view.pitch,
            yaw: self.view.yaw,
            x: self.view.x,
            y: self.view.y,
            z: self.view.z,
        };

       
       let copy_operations = [
        CopyBufferInfo::buffers(self.buffers.face_staging_buffer.clone(), self.buffers.face_buffer.clone()),
        CopyBufferInfo::buffers(self.buffers.running_vertice_staging_buffer.clone(), self.buffers.running_vertice_buffer.clone()),
        CopyBufferInfo::buffers(in_view_staging_buffer.clone(), in_view_buffer.clone()),
        CopyBufferInfo::buffers(self.buffers.vertex_readback_buffer.clone(), self.buffers.vertex_buffer.clone()),
        CopyBufferInfo::buffers(depth_staging_buffer.clone(), depth_buffer.clone()),

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
                clear_value: ClearColorValue::Float([0.0, 0.0, 0.0, 1.0]),
                ..ClearColorImageInfo::image(pixel_image.clone())
            })
                .unwrap()
            .dispatch(work_group_counts)
                .unwrap()
            .copy_image_to_buffer(CopyImageToBufferInfo::image_buffer(
                pixel_image.clone(),
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
}

