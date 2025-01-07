use minifb::{Key, Window, WindowOptions};

const WIDTH: usize = 640;
const HEIGHT: usize = 360;

fn main() {
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

    // Limit to max ~60 fps update rate
    window.set_target_fps(60);

    let mut count = 0;
    while window.is_open() && !window.is_key_down(Key::Escape) {
        count = 0;
        for i in buffer.iter_mut() {
            count += 1;
            let grayscale = (count % 127) + 127 as u32; // Calculate the grayscale value (0-254)
            *i = (grayscale << 16) | (grayscale << 8) | grayscale; // Set RGB to the same value
        }

        // We unwrap here as we want this code to exit if it fails. Real applications may want to handle this in a different way
        window
            .update_with_buffer(&buffer, WIDTH, HEIGHT)
            .unwrap();
    }
}