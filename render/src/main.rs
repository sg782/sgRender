use minifb::{Key, Window, WindowOptions};
use std::time::{Duration, Instant};
use std::sync::{Arc, Mutex};


use crate::world::World;
use crate::view::View;
use crate::renderer::Renderer;
use crate::models::imported::Imported;

use crate::primitives::triangle::Triangle;

pub mod mesh;
pub mod models;
mod plane;
pub mod primitives;
mod world;
mod view;
mod renderer;

const WIDTH: usize = 1200;
const HEIGHT: usize = 1200;

// wud be a cool addition to make it change the fov or something if we move our head closer to the screen

/*
Things to add:
 - turn camera left/right/up/down (relative to view)
 - render faces (fill triangles)

 - add render distance


 - dont render meshes behind camera (make a cubic outline at initialization and if cube is out of render then dont render mesh)


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




    let view = View::new(0.,0.,70.,0.,0.,0.,1.22, WIDTH, HEIGHT);

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




    

    window.set_target_fps(60);


    let mut x0 = 10.;

    // toggle with 'g'
    let mut use_wireframe: bool = true;


    while window.is_open() && !window.is_key_down(Key::Escape) {   



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
                renderer.view.move_forward(-1.);

            }else if window.is_key_down(Key::A){
                x0 += 4.;
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
                renderer.view.rotate_roll(rotation_val);

            }else if window.is_key_down(Key::Y){
                renderer.view.rotate_pitch(rotation_val);
            }else if window.is_key_down(Key::Z){
                renderer.view.rotate_yaw(rotation_val);
            }else if window.is_key_down(Key::G){
                use_wireframe = !use_wireframe;
            }
        }

        
        let now = Instant::now();

     

        

        // let t = Triangle::new(x0,100., 100.,35., 50., 450., 0xFFFFFF);

        // t.draw(&mut buffer, WIDTH as i64,HEIGHT as i64);

        
        renderer.render(&mut buffer, use_wireframe, WIDTH as i64,HEIGHT as i64);

        // We unwrap here as we want this code to exit if it fails. Real applications may want to handle this in a different way
        window
            .update_with_buffer(&buffer, WIDTH, HEIGHT)
            .unwrap();

        println!("{} ms", now.elapsed().as_millis());
    }
}