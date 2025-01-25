
use std::cmp::min;
use std::cmp::max;

use nalgebra::Vector2;

pub struct Line{
    pub x1: f32,
    pub y1: f32,
    pub x2: f32,
    pub y2: f32,
    pub stroke_width: f32,
    pub color: u32,
}



impl Line {

    pub fn new(
        x1: f32,
        y1: f32,
        x2: f32,
        y2: f32,
        stroke_width: f32,
        color: u32,
    ) -> Line {
        
        Line {
            x1, y1, x2, y2, stroke_width, color
        }
    }

    pub fn from_vec (p0: Vector2<f32>, p1: Vector2<f32>, color: u32) -> Line{

        Line {
            x1: p0[0],
            y1: p0[1],
            x2: p1[0],
            y2: p1[1],
            stroke_width: 1.,
            color,
        }
    }



    pub fn draw(&self, buffer: &mut Vec<u32>, screen_width: i64, screen_height: i64){
        // add bounds checks later

        // assume we have integer inputs and strokewidth = 1 rn
        let mut x: i64;
        let mut y: i64;

        let mut index: usize;

        let dy = self.y2 - self.y1;
        let dx = self.x2 - self.x1;


    

        // if it is a single point
        if dy as i64 == 0 && dx as i64 == 0 {
            x = self.x1 as i64;
            y = self.y1 as i64;

            if y<0 || y>=screen_height || x<0 || x>=screen_width {
                return;
            }

            index = (y*(screen_width) + x) as usize;
            buffer[index] = self.color;
            return;
        }


        //determine dominant axis
        if dy.abs() > dx.abs() {
            // y is dominant (or equal), step through y

            let y_range = min(0,dy as i64)..max(0,dy as i64);
            for i in y_range {
                // worry abt strokewidth later

                y = i + self.y1 as i64;

                x = (i as f32 * (dx/dy) + self.x1) as i64;

                if y<0 || y>=screen_height || x<0 || x>=screen_width {
                    return;
                }

                index = ((y*screen_width) + x) as usize;
                buffer[index] = self.color;
            }

        }else{
            // x is dominant, step through x
            let x_range = min(0,dx as i64)..max(0,dx as i64);
            for i in x_range{
                // worry abt strokewidth later

                x = i + self.x1 as i64;

                y = (i as f32 * (dy/dx) + self.y1) as i64;

                if y<0 || y>=screen_height || x<0 || x>=screen_width {
                    return;
                }

                index = ((y*screen_width) + x) as usize;
                buffer[index] = self.color;
            }
        }

    }

    
}