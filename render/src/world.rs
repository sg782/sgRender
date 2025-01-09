use crate::world_element::WorldElement;

pub struct World{
    pub height: i64,
    pub width: i64,
    pub depth: i64,
    pub elements: Vec<WorldElement>,
}

impl World{

    pub fn new(
        height: i64,
        width: i64,
        depth: i64,
    ) -> World {
        let elements: Vec<WorldElement> = Vec::new();
        World {
            height, width, depth, elements,
        }
    }

    // renders as is from the pov of the 
    pub fn render(&self, buffer: &mut Vec<u32>,  screen_width: i64, screen_height: i64){

       buffer[20000] = 0xFFFFFF;
    }

}