use minifb::{Key, Window, WindowOptions};

use crate::line::Line;
use crate::world::World;
use crate::view::View;

mod line;
mod shape;
mod world_element;
mod world;
mod view;
mod renderer;

const WIDTH: usize = 1000;
const HEIGHT: usize = 1200;

fn main() {

    let view = View::new(0.,0.,0.,0.,0.,0.,0.);

    let mut world = World::new(100,100,100);

    
    let mut buffer: Vec<u32> = vec![0; WIDTH * HEIGHT];

    let mut window = Window::new(
        "Test - ESC to exit",
        WIDTH,
        HEIGHT,
        WindowOptions::default(),
    )
    .unwrap_or_else(|e| {
        panic!("{}", e);
    });

    let line = Line::new(10.,10.,10.,10.,1.0,0xFFFFFFF);

    line.draw(&mut buffer,WIDTH as i64,HEIGHT as i64);

    

    // Limit to max ~60 fps update rate
    window.set_target_fps(60);

    while window.is_open() && !window.is_key_down(Key::Escape) {   

        if window.is_key_down(Key::W){
            world.view.move_x(100.);
        }

        world.render(&mut buffer,WIDTH as i64,HEIGHT as i64);
        // We unwrap here as we want this code to exit if it fails. Real applications may want to handle this in a different way
        window
            .update_with_buffer(&buffer, WIDTH, HEIGHT)
            .unwrap();
    }
}