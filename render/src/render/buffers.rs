use vulkano::buffer::Subbuffer;
use nalgebra::Vector2;
use crate::render::structures::{Transformation, BoundingPoint};

use std::f32::INFINITY;

use crate::world::World;
use nalgebra::Vector3;

use nalgebra::Matrix4;
use nalgebra::Vector4;

use vulkano::buffer::{Buffer, BufferCreateInfo, BufferUsage};
use vulkano::memory::allocator::{AllocationCreateInfo, MemoryTypeFilter};


use crate::render::render_information::RenderInformation;

/*
Struct that holds all the relevant buffers that we need

nasty

*/


pub struct Buffers { 
    // vertices
    pub vertex_staging_buffer: Subbuffer<[[f32; 4]]>,
    pub vertex_buffer: Subbuffer<[[f32; 4]]>,
    pub vertex_readback_buffer: Subbuffer<[[f32; 4]]>,
    
    pub unified_vertex_buffer:  Subbuffer<[[f32; 4]]>,

    pub vertex_depth_buffer: Subbuffer<[f32]>,

    // rotation transform
    pub transform_staging_buffers: Vector2<Subbuffer<Transformation>>,
    pub transform_buffer: Subbuffer<Transformation>,

    //faces_buffer
    pub face_staging_buffer: Subbuffer<[[u32; 4]]>,
    pub face_buffer: Subbuffer<[[u32; 4]]>,

    pub face_normal_staging_buffer: Subbuffer<[[f32;4]]>,
    pub face_normal_buffer: Subbuffer<[[f32; 4]]>,

    pub color_staging_buffer: Subbuffer<[u32]>,
    pub color_buffer: Subbuffer<[u32]>,

    // running vertices
    pub running_vertice_staging_buffer: Subbuffer<[u32]>,
    pub running_vertice_buffer: Subbuffer<[u32]>,

    pub bounding_box_staging_buffer: Subbuffer<[BoundingPoint]>,
    pub bounding_box_buffer: Subbuffer<[BoundingPoint]>,

    pub depth_staging_buffer: Subbuffer<[f32]>,
    pub depth_buffer: Subbuffer<[f32]>,

    pub in_view_staging_buffer: Subbuffer<[u32]>,
    pub in_view_buffer: Subbuffer<[u32]>,
    
}

impl Buffers {
    pub fn new(render_information: &RenderInformation, world: &World, screen_width: usize, screen_height: usize) -> Buffers{
        
        let mut raw_vertex_data: Vec<Vector4<f32>> = Vec::new();
        // put all vertex data into one list, and decompose afterward
        for mesh in &world.elements {
            for vertice in mesh.vertices(){
                raw_vertex_data.push(vertice.position);
            }
        }

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

        let vertex_depth_buffer: Subbuffer<[f32]> = Buffer::new_unsized(
            render_information.memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::STORAGE_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
                ..Default::default()
            },
            buffer_size
        ).expect("Failed to create storage buffer!");



        let unified_vertex_buffer: Subbuffer<[[f32; 4]]> = Buffer::new_unsized(
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
        ).expect("Failed to create vertex buffer!");
        
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
            (raw_vertex_data.len() * 3) as u64, // Same size as vertex_buffer
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
        let mut face_normals: Vec<Vector3<f32>> = Vec::new();

        for (i,mesh) in world.elements.iter().enumerate() {
            for face in mesh.faces(){
                face_data.push(Vector4::new(
                    face.vertex_ids[0] as usize,
                    face.vertex_ids[1] as usize, 
                    face.vertex_ids[2] as usize, 
                    i
                ));
                face_normals.push(face.normal);
            }
        }

        let buffer_size: u64 = (face_data.len() * std::mem::size_of::<[usize; 4]>()) as u64; // Total buffer size in bytes

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


        let buffer_size: u64 = (face_normals.len() * std::mem::size_of::<[f32; 4]>()) as u64; // Total buffer size in bytes
        let face_normal_staging_buffer: Subbuffer<[[f32;4]]> = Buffer::from_iter( 
            render_information.memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::TRANSFER_SRC,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            face_normals
            .iter() // Iterate over `Vec<[f32; 4]>`
            .map(|&v| [v[0], v[1], v[2], 0.]),
        ).expect("failed to cretae staging buffer");

        let face_normal_buffer: Subbuffer<[[f32;4]]> = Buffer::new_unsized(
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

        let mut color_data: Vec<u32> = Vec::new();
        for mesh in &world.elements{
            color_data.push(mesh.color());
        }

        let buffer_size: u64 = (color_data.len() * std::mem::size_of::<u32>()) as u64; // Total buffer size in bytes


        let color_staging_buffer: Subbuffer<[u32]> = Buffer::from_iter( 
            render_information.memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::TRANSFER_SRC,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            color_data

        ).expect("failed to cretae staging buffer");

        
        let color_buffer: Subbuffer<[u32]> = Buffer::new_unsized(
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


        let mut depth_vec: Vec<f32> = vec![-INFINITY; (screen_width * screen_height) as usize]; 
        let buffer_size = (depth_vec.len() * std::mem::size_of::<u32>()) as u64; // Total buffer size in bytes


        let depth_staging_buffer: Subbuffer<[f32]> = Buffer::from_iter( 
            render_information.memory_allocator.clone(),
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

                
        let depth_buffer: Subbuffer<[f32]> = Buffer::new_unsized(
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


        let in_vec = vec![0; vertice_indices.len()]; 



        let in_view_staging_buffer: Subbuffer<[u32]> = Buffer::from_iter( 
            render_information.memory_allocator.clone(),
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



        // buffer encapsulation struct
        let buffers = Buffers {
            vertex_staging_buffer,
            vertex_buffer,
            vertex_readback_buffer,

            unified_vertex_buffer,

            vertex_depth_buffer,

            transform_buffer,
            transform_staging_buffers,

            face_staging_buffer,
            face_buffer,

            face_normal_staging_buffer,
            face_normal_buffer,

            color_staging_buffer,
            color_buffer,
            
            running_vertice_staging_buffer,
            running_vertice_buffer,

            bounding_box_staging_buffer,
            bounding_box_buffer,

            depth_staging_buffer,
            depth_buffer,

            in_view_staging_buffer,
            in_view_buffer,


        };

        buffers

    }
}