use minifb::{Key, Window, WindowOptions};
use nalgebra::Vector3;
use std::time::Instant;


use std::env;

use crate::world::World;
use crate::view::View;
use crate::renderer::Renderer;

pub mod lighting;
pub mod shaders;
pub mod mesh;
pub mod models;
mod plane;
pub mod primitives;
mod world;
mod view;
mod renderer;

const WIDTH: usize = 1200;
const HEIGHT: usize = 600;



// wud be a cool addition to make it change the fov or something if we move our head closer to the screen

/*
Things to add:
 - turn camera left/right/up/down (relative to view)
 - render faces (fill triangles)
 - dynamically sized (at initialization) buffers for rendering
 - add render distance


*/


/*
controls

w - move forward
s - move backward

a - move left
d - move right

q - move up
e - move down

z - rotate positive around z axis
shift z - rotate negative around z axis

x - rotate positive around x axis
shift x - rotate negative around x axis

y - rotate positie around y axis
shift y - rotate negatie around y axis

*/



fn main() {

    env::set_var("RUST_BACKTRACE", "1");

    let view = View::new(0.,0.,50.,0.,0.,0.,1., WIDTH, HEIGHT);
    let world = World::new();
    let mut renderer = Renderer::new(world,view, WIDTH, HEIGHT);


    let mut window = Window::new(
        "Test - ESC to exit",
        WIDTH,
        HEIGHT,
        WindowOptions::default(),
    )
    .unwrap_or_else(|e| {
        panic!("{}", e);
    });

    
    let mut light: Vector3<f32> = Vector3::new(1.,1.,1.);
    

    window.set_target_fps(500);

    // toggle with 'g'
    let mut use_wireframe: bool = true;


    while window.is_open() && !window.is_key_down(Key::Escape) {   
        let movement_val = 0.4;

        // random keys for testing
        // will clean up later
        let rotation_val = 0.03;
        if window.is_key_down(Key::LeftShift){
            if window.is_key_down(Key::W){
            }else if window.is_key_down(Key::A){
            }else if window.is_key_down(Key::S){
            }else if window.is_key_down(Key::D){

            }else if window.is_key_down(Key::X){
                renderer.view.rotate_roll(-rotation_val);
            }else if window.is_key_down(Key::Y){
                renderer.view.rotate_pitch(-rotation_val);
            }else if window.is_key_down(Key::Z){
                renderer.view.rotate_yaw(-rotation_val);
            }
        }else{
            if window.is_key_down(Key::W){
                // testing
                renderer.view.move_forward(-movement_val);

            }else if window.is_key_down(Key::A){
                renderer.view.move_side(-movement_val);

            }else if window.is_key_down(Key::S){
                renderer.view.move_forward(movement_val);
            
            }else if window.is_key_down(Key::Q){
                renderer.view.move_vertical(-movement_val)
            }else if window.is_key_down(Key::E){
                renderer.view.move_vertical(movement_val)

            }else if window.is_key_down(Key::D){
                renderer.view.move_side(movement_val);

            }else if window.is_key_down(Key::X){
                renderer.view.rotate_roll(rotation_val);

            }else if window.is_key_down(Key::Y){
                renderer.view.rotate_pitch(rotation_val);
            }else if window.is_key_down(Key::Z){
                renderer.view.rotate_yaw(rotation_val);
            }
        }

        
        let now = Instant::now();

     

        

        // let t = Triangle::new(x0,100., 100.,35., 50., 450., 0xFFFFFF);

        // t.draw(&mut buffer, WIDTH as i64,HEIGHT as i64);


        //println!("{} {} {}", renderer.view.x, renderer.view.y, renderer.view.z);

        
        renderer.render(&mut window);

        // We unwrap here as we want this code to exit if it fails. Real applications may want to handle this in a different way
        // window
        //     .update_with_buffer(&pixel_buffer, WIDTH, HEIGHT)
        //     .unwrap();

        println!("{} ms || ~{} fps", now.elapsed().as_millis(), 1000 / now.elapsed().as_millis());
    }
}