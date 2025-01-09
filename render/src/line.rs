use crate::WIDTH;


pub struct Line{
    pub x1: f64,
    pub y1: f64,
    pub x2: f64,
    pub y2: f64,
    pub stroke_width: f64,
    pub color: u32,
}

impl Line {

    pub fn new(
        x1: f64,
        y1: f64,
        x2: f64,
        y2: f64,
        stroke_width: f64,
        color: u32,
    ) -> Line {
        
        Line {
            x1, y1, x2, y2, stroke_width, color
        }

    }
    pub fn draw(&self, buffer: &mut Vec<u32>, screen_width: i64, screen_height: i64){
        // add bounds checks later

        // assume we have integer inputs and strokewidth = 1 rn

        let mut x_float: f64;
        let mut y_float: f64;
        let mut index: i64;

        let mut x: i64;
        let mut y: i64;

        let mut index: usize;

        let dy = self.y2-self.y1;
        let dx = self.x2-self.x1;


        if dy as i64 == 0 && dx as i64 == 0 {
            x = self.x1 as i64;
            y = self.y1 as i64;

            
            if y<0 || y>screen_height{
                return;
            }
            if x<0 || x>screen_width{
                return;
            }

            index = (y*screen_width + x) as usize;
            buffer[index] = self.color;
            return;
        }


        //determine dominant axis
        if dy > dx {
            // y is dominant (or equal), step through y
            for i in 0..dy as i64{
                // worry abt strokewidth later

                y = i + self.y1 as i64;

                x = (i as f64 * (dx/dy) + self.x1) as i64;

                if y<0 || y>screen_height{
                    continue;
                }
                if x<0 || x>screen_width{
                    continue;
                }

                index = ((y*screen_width) + x) as usize;
                buffer[index] = self.color;
            }

        }else{
            // x is dominant, step through x
            for i in 0..dx as i64{
                // worry abt strokewidth later

                x = i + self.x1 as i64;

                y = (i as f64 * (dy/dx) + self.y1) as i64;

                if y<0 || y>screen_height{
                    continue;
                }
                if x<0 || x>screen_width{
                    continue;
                }

                index = ((y*screen_width) + x) as usize;
                buffer[index] = self.color;
            }
        }

        


    }

}