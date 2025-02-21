use vulkano::buffer::Subbuffer;
use nalgebra::Vector2;
use crate::render::structures::{Transformation, BoundingPoint};

use std::f32::INFINITY;

use vulkano::image::{Image, ImageUsage, view::ImageView};
use vulkano::format::Format;

use crate::lighting::point_light::PointLight;


use crate::world::World;
use nalgebra::Vector3;
use crate::view::View;

use crate::plane::Plane;

use std::sync::Arc;

use nalgebra::Matrix4;
use nalgebra::Vector4;
use nalgebra::Vector5;

use vulkano::buffer::{Buffer, BufferCreateInfo, BufferUsage};
use vulkano::memory::allocator::{AllocationCreateInfo, MemoryTypeFilter};


use crate::render::render_information::RenderInformation;

use vulkano::image::ImageCreateInfo;

use vulkano::image::ImageType;

use super::structures::FrustumFaces;


/*
Struct that holds all the relevant buffers that we need

nasty

*/


pub struct Buffers { 
    // vertices
    pub vertex_staging_buffer: Subbuffer<[[f32; 4]]>,
    pub vertex_buffer: Subbuffer<[[f32; 4]]>,
    pub vertex_readback_buffer: Subbuffer<[[f32; 4]]>,
    
    pub transformed_vertex_buffer:  Subbuffer<[[f32; 4]]>,

    pub vertex_depth_buffer: Subbuffer<[f32]>,

    // rotation transform
    pub transform_staging_buffers: Vector2<Subbuffer<Transformation>>,
    pub transform_buffer: Subbuffer<Transformation>,


    //faces_buffer
    pub face_staging_buffer: Subbuffer<[u32]>,
    pub face_buffer: Subbuffer<[u32]>,

    pub face_normal_staging_buffer: Subbuffer<[[f32;4]]>,
    pub face_normal_buffer: Subbuffer<[[f32; 4]]>,

    pub color_staging_buffer: Subbuffer<[u32]>,
    pub color_buffer: Subbuffer<[u32]>,

    // running vertices
    pub running_vertice_staging_buffer: Subbuffer<[u32]>,
    pub running_vertice_buffer: Subbuffer<[u32]>,

    pub bounding_box_staging_buffer: Subbuffer<[BoundingPoint]>,
    pub bounding_box_buffer: Subbuffer<[BoundingPoint]>,

    pub depth_staging_buffer: Subbuffer<[u32]>,
    pub depth_buffer: Subbuffer<[u32]>,

    pub in_view_staging_buffer: Subbuffer<[u32]>,
    pub in_view_buffer: Subbuffer<[u32]>,

    pub pixel_image: Arc<Image>,
    pub pixel_image_view: Arc<ImageView>,
    pub output_img_buf: Subbuffer<[u32]>,

    pub frustum_faces_staging_buffers: Vector2<Subbuffer<FrustumFaces>>,
    pub frustum_faces_buffer: Subbuffer<FrustumFaces>,

    pub point_light_staging_buffer: Subbuffer<[[f32;4]]>,
    pub point_light_buffer: Subbuffer<[[f32;4]]>,

    
}

impl Buffers {
    pub fn new(render_information: &RenderInformation, world: &World, view: &View, screen_width: usize, screen_height: usize) -> Buffers{
        
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

        ).expect("failed to create staging buffer");

        
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



        let transformed_vertex_buffer: Subbuffer<[[f32; 4]]> = Buffer::new_unsized(
            render_information.memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::TRANSFER_DST | BufferUsage::TRANSFER_SRC | BufferUsage::STORAGE_BUFFER,  // Transfer destination for readback
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



        let mut face_data: Vec<Vector5<u32>> = Vec::new();
        let mut face_normals: Vec<Vector3<f32>> = Vec::new();

        for (i,mesh) in world.elements.iter().enumerate() {
            for face in mesh.faces(){
                face_data.push(Vector5::new(
                    face.vertex_ids[0] as u32,
                    face.vertex_ids[1] as u32, 
                    face.vertex_ids[2] as u32, 
                    face.vertex_ids[3] as u32,
                    i as u32
                ));
                face_normals.push(face.normal);
            }
        }

        let flat_faces: Vec<u32> = face_data.iter().flat_map(|v| [v.x,v.y,v.z,v.w, v[4]]).collect();

        let buffer_size: u64 = (face_data.len() * std::mem::size_of::<[usize; 4]>()) as u64; // Total buffer size in bytes

        let face_staging_buffer: Subbuffer<[u32]> = Buffer::from_iter( 
            render_information.memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::TRANSFER_SRC,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            flat_faces

        ).expect("failed to create staging buffer");

        
        let face_buffer: Subbuffer<[u32]> = Buffer::new_unsized(
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
        ).expect("failed to create staging buffer");

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

        ).expect("failed to create staging buffer");

        
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

        ).expect("failed to create staging buffer");

        
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

        let mut depth_vec_a: Vec<u32> = vec![u32::MAX; (screen_width * screen_height) as usize];

        let buffer_size = (depth_vec.len() * std::mem::size_of::<u32>()) as u64; // Total buffer size in bytes


        let depth_staging_buffer: Subbuffer<[u32]> = Buffer::from_iter( 
            render_information.memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::TRANSFER_SRC,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            depth_vec_a.iter().cloned(),

        ).expect("failed to create depth buffer");

                
        let depth_buffer: Subbuffer<[u32]> = Buffer::new_unsized(
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

        ).expect("failed to create staging buffer");

        
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


        let pixel_image = Image::new(
            render_information.memory_allocator.clone(),
            ImageCreateInfo {
                image_type: ImageType::Dim2d,
                format: Format::R8G8B8A8_UNORM,
                extent: [screen_width as u32,screen_height as u32, 1],
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

        let pixel_image_size = screen_width * screen_height * 4; // RGBA, 4 bytes per pixel
        let output_img_buf: Subbuffer<[u32]> = Buffer::new_slice(
            render_information.memory_allocator.clone(),
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


        let frustum_faces = &view.frustum_faces;

        

        let frustum_faces_staging_buffer_1:Subbuffer<FrustumFaces> = Buffer::from_data(
            render_information.memory_allocator.clone(),
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

        let frustum_faces_staging_buffer_2:Subbuffer<FrustumFaces> = Buffer::from_data(
            render_information.memory_allocator.clone(),
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

        let frustum_faces_staging_buffers:Vector2<Subbuffer<FrustumFaces>> = Vector2::new(
            frustum_faces_staging_buffer_1,
            frustum_faces_staging_buffer_2,
        );
        


        let frustum_faces_buffer: Subbuffer<FrustumFaces> = Buffer::new_unsized(
            render_information.memory_allocator.clone(),
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


        let mut raw_point_light_data: Vec<Vector4<f32>> = Vec::new();
        // put all vertex data into one list, and decompose afterward
        for point_light in &world.lights.point_lights {
            raw_point_light_data.push(point_light.as_vec_4());
        }


        let buffer_size = (raw_point_light_data.len() * std::mem::size_of::<[f32; 4]>()) as u64; // Total buffer size in bytes

        let point_light_staging_buffer: Subbuffer<[[f32; 4]]> = Buffer::from_iter( 
            render_information.memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::TRANSFER_SRC,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            raw_point_light_data
            .iter() // Iterate over `Vec<[f32; 4]>`
            .map(|&v| [v[0] as f32, v[1] as f32, v[2] as f32, v[3] as f32]),

        ).expect("failed to create staging buffer");
        let point_light_buffer: Subbuffer<[[f32; 4]]> = Buffer::new_unsized(
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

            transformed_vertex_buffer,

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

            pixel_image,
            pixel_image_view,
            output_img_buf,

            frustum_faces_staging_buffers,
            frustum_faces_buffer,

            point_light_buffer,
            point_light_staging_buffer,
        };

        buffers

    }
}