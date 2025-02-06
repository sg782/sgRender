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


use nalgebra::Vector4;







pub struct RenderInformation {
    pub device: Arc<Device>,
    pub queue: Arc<Queue>,
    pub command_buffer_allocator: StandardCommandBufferAllocator,
    pub memory_allocator: Arc<StandardMemoryAllocator>,
    pub vertex_compute_pipeline: Arc<ComputePipeline>,
    pub line_draw_compute_pipeline: Arc<ComputePipeline>,
    pub in_view_compute_pipeline: Arc<ComputePipeline>,
    pub triangle_draw_compute_pipeline: Arc<ComputePipeline>,
}

impl RenderInformation {

    fn load_shader(device: Arc<vulkano::device::Device>, path: &str) -> Arc<ShaderModule> {
        let mut file = File::open(path).expect("Failed to open shader file");
        let mut shader_bytes = Vec::new();
        file.read_to_end(&mut shader_bytes).unwrap();

        unsafe { ShaderModule::from_bytes(device.clone(), &shader_bytes) }
            .expect("Failed to create shader module")
    }


    pub fn fill_vertex_buffer(&self) {
        
    }


    pub fn new(device: Arc<Device>, queue: Arc<Queue>, raw_vertex_data: &Vec<Vector4<f32>>) -> Self {

        // Memory Allocator (Reuse across frames)
        let memory_allocator = Arc::new(StandardMemoryAllocator::new_default(device.clone()));

        // Command Buffer Allocator (Reuse for submitting commands)
        let command_buffer_allocator = StandardCommandBufferAllocator::new(device.clone(), Default::default());

        // Create Buffer (Keep the same buffer across frames)

        let vertex_shader = RenderInformation::load_shader(device.clone(), "src/shaders/vertices.spv");
        let vertex_entry_point = vertex_shader.entry_point("main").unwrap();
        let stage = PipelineShaderStageCreateInfo::new(vertex_entry_point);
        let layout = PipelineLayout::new(
            device.clone(),
            PipelineDescriptorSetLayoutCreateInfo::from_stages([&stage])
                .into_pipeline_layout_create_info(device.clone())
                .unwrap()
        )
        .unwrap();

        let vertex_compute_pipeline = ComputePipeline::new(
            device.clone(),
            None,
            ComputePipelineCreateInfo::stage_layout(stage, layout),
        )
        .expect("failed to create compute pipeline");


        let line_draw_shader = RenderInformation::load_shader(device.clone(), "src/shaders/line_draw.spv");
        let line_draw_entry_point = line_draw_shader.entry_point("main").unwrap();
        let line_stage = PipelineShaderStageCreateInfo::new(line_draw_entry_point);


        

        let line_layout = PipelineLayout::new(
            device.clone(),
            PipelineDescriptorSetLayoutCreateInfo::from_stages([&line_stage])
                .into_pipeline_layout_create_info(device.clone())
                .unwrap()
        )
        .unwrap();



        let line_draw_compute_pipeline = ComputePipeline::new(
            device.clone(),
            None,
            ComputePipelineCreateInfo::stage_layout(line_stage, line_layout),
        )
        .expect("failed to create compute pipeline");


        let in_view_shader = RenderInformation::load_shader(device.clone(), "src/shaders/in_view.spv");
        let in_view_entry_point = in_view_shader.entry_point("main").unwrap();
        let in_view_stage = PipelineShaderStageCreateInfo::new(in_view_entry_point);


        

        let in_view_layout = PipelineLayout::new(
            device.clone(),
            PipelineDescriptorSetLayoutCreateInfo::from_stages([&in_view_stage])
                .into_pipeline_layout_create_info(device.clone())
                .unwrap()
        )
        .unwrap();


        let in_view_compute_pipeline = ComputePipeline::new(
            device.clone(),
            None,
            ComputePipelineCreateInfo::stage_layout(in_view_stage, in_view_layout),
        )
        .expect("failed to create compute pipeline");


        let triangle_draw_shader = RenderInformation::load_shader(device.clone(), "src/shaders/triangle_draw.spv");
        let triangle_draw_entry_point = triangle_draw_shader.entry_point("main").unwrap();
        let triangle_draw_stage = PipelineShaderStageCreateInfo::new(triangle_draw_entry_point);


        let triangle_draw_layout = PipelineLayout::new(
            device.clone(),
            PipelineDescriptorSetLayoutCreateInfo::from_stages([&triangle_draw_stage])
                .into_pipeline_layout_create_info(device.clone())
                .unwrap()
        )
        .unwrap();



        let triangle_draw_compute_pipeline = ComputePipeline::new(
            device.clone(),
            None,
            ComputePipelineCreateInfo::stage_layout(triangle_draw_stage, triangle_draw_layout),
        )
        .expect("failed to create compute pipeline");


        Self {
            device,
            queue,
            command_buffer_allocator,
            memory_allocator,
            vertex_compute_pipeline,
            line_draw_compute_pipeline,
            in_view_compute_pipeline,
            triangle_draw_compute_pipeline,
        }
    }

}