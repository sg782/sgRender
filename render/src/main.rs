use std::f32::consts::PI;

use minifb::{Key, Window, WindowOptions};

use crate::world::World;
use crate::view::View;
use crate::renderer::Renderer;

mod line;
mod world;
mod view;
mod renderer;
mod point;
mod mesh;
mod face;

const WIDTH: usize = 700;
const HEIGHT: usize = 700;

// wud be a cool addition to make it change the fov or something if we move our head closer to the screen

fn main() {

    let view = View::new(0.,0.,20.,0.,0.,0.,70.);

    let world = World::new(HEIGHT as i64,WIDTH as i64,1000);

    let mut renderer = Renderer::new(world,view);


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

    // let line = Line::new(10.,10.,10.,10.,1.0,0xFFFFFFF);

    // line.draw(&mut buffer,WIDTH as i64,HEIGHT as i64);

    

    // Limit to max ~60 fps update rate
    window.set_target_fps(20);

    while window.is_open() && !window.is_key_down(Key::Escape) {   


        let rotation_val = 0.001;
        if window.is_key_down(Key::LeftShift){
            if window.is_key_down(Key::W){
            }else if window.is_key_down(Key::A){
            }else if window.is_key_down(Key::S){
            }else if window.is_key_down(Key::D){

            }else if window.is_key_down(Key::X){
                renderer.view.rotate_roll(PI as f64 / -500.);
            }else if window.is_key_down(Key::Y){
                renderer.view.rotate_pitch(PI as f64 / -500.);
            }else if window.is_key_down(Key::Z){
                renderer.view.rotate_yaw(PI as f64 / -500.);
            }
        }else{
            if window.is_key_down(Key::W){
                // testing
                renderer.view.move_forward(-1.);

            }else if window.is_key_down(Key::A){
                renderer.view.move_side(-5.);

            }else if window.is_key_down(Key::S){
                renderer.view.move_forward(1.);
            
            }else if window.is_key_down(Key::Q){
                renderer.view.move_vertical(-5.)
            }else if window.is_key_down(Key::E){
                renderer.view.move_vertical(5.)

            }else if window.is_key_down(Key::D){
                renderer.view.move_side(5.);

            }else if window.is_key_down(Key::X){
                renderer.view.rotate_roll(PI as f64 / 100.);
            }else if window.is_key_down(Key::Y){
                renderer.view.rotate_pitch(PI as f64 / 100.);
            }else if window.is_key_down(Key::Z){
                renderer.view.rotate_yaw(PI as f64 / 100.);
            }
        }

        renderer.render(&mut buffer,WIDTH as i64,HEIGHT as i64);

        // We unwrap here as we want this code to exit if it fails. Real applications may want to handle this in a different way
        window
            .update_with_buffer(&buffer, WIDTH, HEIGHT)
            .unwrap();
    }
}