use std::sync::Arc;

use vulkano::VulkanLibrary;
use vulkano::instance::{Instance, InstanceCreateFlags, InstanceCreateInfo};
use vulkano::device::QueueFlags;
use vulkano::device::{Features};

use vulkano::device::{ DeviceCreateInfo, QueueCreateInfo};

use vulkano::descriptor_set::{PersistentDescriptorSet, WriteDescriptorSet};


use vulkano::device::{Device, Queue};
use vulkano::memory::allocator::{StandardMemoryAllocator};
use vulkano::command_buffer::allocator::StandardCommandBufferAllocator;
use std::fs::File;
use std::io::Read;
use vulkano::shader::ShaderModule;
use vulkano::pipeline::{ComputePipeline, PipelineShaderStageCreateInfo};
use vulkano::pipeline::layout::PipelineDescriptorSetLayoutCreateInfo;
use vulkano::pipeline::layout::{PipelineLayout, PipelineLayoutCreateInfo};
use vulkano::pipeline::compute::ComputePipelineCreateInfo;
use vulkano::descriptor_set::allocator::StandardDescriptorSetAllocator;
use vulkano::pipeline::Pipeline;


pub struct RenderInformation {
    pub device: Arc<Device>,
    pub queue: Arc<Queue>,
    pub command_buffer_allocator: StandardCommandBufferAllocator,
    pub memory_allocator: Arc<StandardMemoryAllocator>,
    pub vertex_compute_pipeline: Arc<ComputePipeline>,
    pub line_draw_compute_pipeline: Arc<ComputePipeline>,
    pub in_view_compute_pipeline: Arc<ComputePipeline>,
    pub triangle_draw_compute_pipeline: Arc<ComputePipeline>,
    pub sum_faces_by_tile_compute_pipeline: Arc<ComputePipeline>,

    pub work_group_counts: [u32;3],
}

impl RenderInformation {

    fn load_shader(device: Arc<vulkano::device::Device>, path: &str) -> Arc<ShaderModule> {
        let mut file = File::open(path).expect("Failed to open shader file");
        let mut shader_bytes = Vec::new();
        file.read_to_end(&mut shader_bytes).unwrap();

        unsafe { ShaderModule::from_bytes(device.clone(), &shader_bytes) }
            .expect("Failed to create shader module")
    }


    pub fn new() -> Self {

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

        let features = Features {
            shader_float64: true, // Enable double-precision floating point support
            ..Features::empty()    // Keep other features disabled unless needed
        };
        
    
        let (device, mut queues) = Device::new(
            physical_device,
            DeviceCreateInfo {
                enabled_features: features,
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

        // Memory Allocator (Reuse across frames)
        let memory_allocator = Arc::new(StandardMemoryAllocator::new_default(device.clone()));

        // Command Buffer Allocator (Reuse for submitting commands)
        let command_buffer_allocator = StandardCommandBufferAllocator::new(device.clone(), Default::default());

        let vertex_shader_path = "src/shaders/vertices.spv";
        let vertex_compute_pipeline = RenderInformation::create_compute_pipeline(device.clone(), &vertex_shader_path);

        let line_draw_shader_path = "src/shaders/line_draw.spv";
        let line_draw_compute_pipeline = RenderInformation::create_compute_pipeline(device.clone(), &line_draw_shader_path);

        let in_view_shader_path = "src/shaders/in_view.spv";
        let in_view_compute_pipeline = RenderInformation::create_compute_pipeline(device.clone(), in_view_shader_path);


        let triangle_draw_shader_path = "src/shaders/triangle_draw.spv";
        let triangle_draw_compute_pipeline = RenderInformation::create_compute_pipeline(device.clone(), &triangle_draw_shader_path);
        
        let sum_faces_by_tile_shader_path = "src/shaders/sum_face_intersections.spv";
        let sum_faces_by_tile_compute_pipeline = RenderInformation::create_compute_pipeline(device.clone(),&sum_faces_by_tile_shader_path);

        // let calculate_tile_offsets_shader_path = "";
        // let calculate_tile_offsets_compute_pipeline = RenderInformation::create_compute_pipeline(device.clone(),&calculate_tile_offsets_shader_path);

        // let sort_faces_by_tile_shader_path = "";
        // let sort_faces_by_tile_compute_pipeline =  RenderInformation::create_compute_pipeline(device.clone(),&sort_faces_by_tile_shader_path);


        let work_group_counts = [8192,1,1];


        Self {
            device,
            queue,
            command_buffer_allocator,
            memory_allocator,
            vertex_compute_pipeline,
            line_draw_compute_pipeline,
            in_view_compute_pipeline,
            triangle_draw_compute_pipeline,
            sum_faces_by_tile_compute_pipeline,
            work_group_counts,
        }
    }

    fn create_compute_pipeline(device: Arc<Device>, path: &str) -> Arc<ComputePipeline>{
        let shader = RenderInformation::load_shader(device.clone(), path);
        let entry_point = shader.entry_point("main").unwrap();
        let stage = PipelineShaderStageCreateInfo::new(entry_point);


        let layout = PipelineLayout::new(
            device.clone(),
            PipelineDescriptorSetLayoutCreateInfo::from_stages([&stage])
                .into_pipeline_layout_create_info(device.clone())
                .unwrap()
        )
        .unwrap();

        let compute_pipeline = ComputePipeline::new(
            device.clone(),
            None,
            ComputePipelineCreateInfo::stage_layout(stage, layout),
        )
        .expect("failed to create compute pipeline");

       compute_pipeline

    }


    // fn create_in_view_descriptor_set(device: Arc<Device>, compute_pipeline: Arc<ComputePipeline>) -> Arc<PersistentDescriptorSet>{
    //     let descriptor_set_allocator =
    //     StandardDescriptorSetAllocator::new(device.clone(), Default::default());

    //     let pipeline_layout = compute_pipeline.layout();
    //     let descriptor_set_layouts = pipeline_layout.set_layouts();

    //     let descriptor_set_layout_index = 0;
    //     let descriptor_set_layout = descriptor_set_layouts
    //         .get(descriptor_set_layout_index)
    //         .unwrap();

    //     let descriptor_set = PersistentDescriptorSet::new(
    //         &descriptor_set_allocator,
    //         descriptor_set_layout.clone(),
    //         [
    //             WriteDescriptorSet::buffer(0, buffers.bounding_box_buffer.clone()),
    //             WriteDescriptorSet::buffer(1,buffers.frustum_faces_buffer.clone()),
    //             WriteDescriptorSet::buffer(2, buffers.in_view_buffer.clone()),
    //             ], 
    //         [],
    //     )
    //     .unwrap();   

    //     descriptor_set

    // }   
}