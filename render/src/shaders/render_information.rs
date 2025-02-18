use std::sync::Arc;

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





pub struct RenderInformation {
    pub device: Arc<Device>,
    pub queue: Arc<Queue>,
    pub command_buffer_allocator: StandardCommandBufferAllocator,
    pub memory_allocator: Arc<StandardMemoryAllocator>,
    pub vertex_compute_pipeline: Arc<ComputePipeline>,
    pub line_draw_compute_pipeline: Arc<ComputePipeline>,
    pub in_view_compute_pipeline: Arc<ComputePipeline>,
    pub triangle_draw_compute_pipeline: Arc<ComputePipeline>,
    pub sort_faces_by_tile_compute_pipeline: Arc<ComputePipeline>,

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




    pub fn new(device: Arc<Device>, queue: Arc<Queue>) -> Self {

        // Memory Allocator (Reuse across frames)
        let memory_allocator = Arc::new(StandardMemoryAllocator::new_default(device.clone()));

        // Command Buffer Allocator (Reuse for submitting commands)
        let command_buffer_allocator = StandardCommandBufferAllocator::new(device.clone(), Default::default());
        
        let vertex_shader_path = "src/shaders/vertices.spv";
        let vertex_compute_pipeline = RenderInformation::create_compute_pipeline(device.clone(), &vertex_shader_path);

        let line_draw_shader_path = "src/shaders/sort_faces_by_tile.spv";
        let line_draw_compute_pipeline = RenderInformation::create_compute_pipeline(device.clone(), &line_draw_shader_path);

        let in_view_shader_path = "src/shaders/in_view.spv";
        let in_view_compute_pipeline = RenderInformation::create_compute_pipeline(device.clone(), in_view_shader_path);

        let triangle_draw_shader_path = "src/shaders/triangle_draw.spv";
        let triangle_draw_compute_pipeline = RenderInformation::create_compute_pipeline(device.clone(), &triangle_draw_shader_path);
        
        let sort_faces_by_tile_shader_path = "src/shaders/sort_faces_by_tile.spv";
        let sort_faces_by_tile_compute_pipeline = RenderInformation::create_compute_pipeline(device.clone(),&sort_faces_by_tile_shader_path);


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
            sort_faces_by_tile_compute_pipeline,
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

}