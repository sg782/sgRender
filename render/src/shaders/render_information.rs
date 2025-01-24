use std::sync::Arc;

use vulkano::device::{Device, Queue};
use vulkano::buffer::{Buffer, BufferCreateInfo, BufferUsage, Subbuffer};
use vulkano::memory::allocator::{StandardMemoryAllocator,AllocationCreateInfo, MemoryTypeFilter};
use vulkano::command_buffer::allocator::StandardCommandBufferAllocator;

pub struct RenderInformation {
    pub device: Arc<Device>,
    pub queue: Arc<Queue>,
    pub command_buffer_allocator: StandardCommandBufferAllocator,
    pub memory_allocator: Arc<StandardMemoryAllocator>,
}

impl RenderInformation {

    pub fn new(device: Arc<Device>, queue: Arc<Queue>) -> Self {

        // Memory Allocator (Reuse across frames)
        let memory_allocator = Arc::new(StandardMemoryAllocator::new_default(device.clone()));

        // Command Buffer Allocator (Reuse for submitting commands)
        let command_buffer_allocator = StandardCommandBufferAllocator::new(device.clone(), Default::default());

        // Create Buffer (Keep the same buffer across frames)

        Self {
            device,
            queue,
            command_buffer_allocator,
            memory_allocator,
        }
    }

}