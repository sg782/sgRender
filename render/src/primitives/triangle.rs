
use std::cmp::min;
use std::cmp::max;
use std::f64::INFINITY;
use std::prelude::v1;

use nalgebra::Vector4;
use nalgebra::Vector3;
use nalgebra::Vector2;

pub struct Triangle{
    pub v1: Vector2<f64>,
    pub v2: Vector2<f64>,
    pub v3: Vector2<f64>,
    pub depth: f64,
    pub color: u32,
}



impl Triangle {

    // always fill (for now)

    pub fn new(
        x0: f64,
        y0: f64,
        x1: f64,
        y1: f64,
        x2: f64,
        y2: f64,
        color: u32,
    ) -> Triangle {

        let points: Vector3<Vector2<f64>> = Vector3::new(
            Vector2::new(x0,y0),
            Vector2::new(x1,y1),
            Vector2::new(x2,y2),
        );
        
        let out = Triangle::sort_vertices(&points);

        Triangle {
            v1: out[0],
            v2: out[1],
            v3: out[2],
            depth: 0.,
            color,
        }
    }

    pub fn from_vec3_list(
        v: &Vector3<Vector2<f64>>,
        color: u32,
    ) -> Triangle {

        let out = Triangle::sort_vertices(&v);
        

        Triangle {
            v1: out[0],
            v2: out[1],
            v3: out[2],
            depth: 0.,
            color,
        }
    }

    pub fn from_vec4_list(
        v: &Vector4<Vector2<f64>>,
        color: u32,
    ) -> Triangle {

        let vertices = v.xyz(); //gets first 3 entries of v

        let out = Triangle::sort_vertices(&vertices);
        

        Triangle {
            v1: out[0],
            v2: out[1],
            v3: out[2],
            depth: v[3][0],
            color,
        }
    }


    pub fn sort_vertices(v: &Vector3<Vector2<f64>>) -> Vector3<Vector2<f64>> {

        // in order of y
        // v1.y <= v2.y <= v3.y
        let v1: Vector2<f64>; 
        let v2: Vector2<f64>;
        let v3: Vector2<f64>;

        // find top
        let mut v1_idx = 3;
        let mut min_y = INFINITY;
        for i in 0..3 {
            if v[i][1] < min_y {
                v1_idx = i;
                min_y = v[i][1];
            }
        }

        let mut v3_idx = 3;
        let mut max_y = - INFINITY;
        // find bottom
        for i in 0..3 {
            if i == v1_idx {continue}

            if v[i][1] > max_y {
                v3_idx = i;
                max_y = v[i][1];
            }
        }

        let v2_idx = 3 - v1_idx - v3_idx;

        v1 = v[v1_idx];
        v2 = v[v2_idx];
        v3 = v[v3_idx];


        let out: Vector3<Vector2<f64>> = Vector3::new(
            v1, v2, v3,
        );


        out
    }



    pub fn draw(&self, pixel_buffer: &mut Vec<u32>, depth_buffer: &mut Vec<f64>, screen_width: i64, screen_height: i64){


        /*
        https://www.sunshine2k.de/coding/java/TriangleRasterization/TriangleRasterization.html
         */



        // points are now sorted from highest to lowest
        // can proceed with alg

        if self.v2[1] == self.v3[1] {
            self.draw_flat_bottom_triangle(pixel_buffer, depth_buffer, self.v1, self.v2, self.v3, screen_width, screen_height);
        }else if self.v1[1] == self.v2[1] {
            self.draw_flat_top_triangle(pixel_buffer, depth_buffer, self.v1, self.v2, self.v3, screen_width, screen_height);
        }else{

            // calculating intersection point for both triangles
            let v4_x = self.v1[0] + ((self.v2[1] - self.v1[1]) / (self.v3[1] - self.v1[1])) * (self.v3[0] - self.v1[0]);
            let v4: Vector2<f64> = Vector2::new(v4_x,self.v2[1]);

            self.draw_flat_bottom_triangle(pixel_buffer, depth_buffer,self.v1, self.v2, v4, screen_width, screen_height); 
            self.draw_flat_top_triangle(pixel_buffer, depth_buffer,self.v2, v4, self.v3, screen_width, screen_height);

        }


    }

    pub fn draw_flat_bottom_triangle(&self, pixel_buffer: &mut Vec<u32>, depth_buffer: & mut Vec<f64>, v1: Vector2<f64>, v2: Vector2<f64>, v3: Vector2<f64>, screen_width: i64, screen_height: i64){
        let inv_slope_1 = (v2[0] - v1[0]) / (v2[1] - v1[1]);
        let inv_slope_2 = (v3[0] - v1[0]) / (v3[1]- v1[1]);

        let mut cur_x_1 = v1[0];
        let mut cur_x_2 = v1[0];

        for i in (v1[1] as i64)..(v2[1] as i64){

            for j in (cur_x_1.min(cur_x_2) as i64)..(cur_x_1.max(cur_x_2)as i64) {
                // draw line

                if i<0 || i>=screen_height || j<0 || j>=screen_width {
                    return; 
                    //
                    // make a proper triangle clipping alg
                }
    
                
                let index = (i*(screen_width) + j) as usize;

                // a more negative depth means further
                if depth_buffer[index] <= self.depth {
                    depth_buffer[index] = self.depth;
                    pixel_buffer[index] = self.color;

                }


            }

            cur_x_1 += inv_slope_1;
            cur_x_2 += inv_slope_2;

        }
    }

    pub fn draw_flat_top_triangle(&self, pixel_buffer: &mut Vec<u32>, depth_buffer: &mut Vec<f64>, v1: Vector2<f64>, v2: Vector2<f64>, v3: Vector2<f64>, screen_width: i64, screen_height: i64){
        let inv_slope_1 = (v3[0] - v1[0]) / (v3[1] - v1[1]);
        let inv_slope_2 = (v3[0] - v2[0]) / (v3[1]- v2[1]);

        let mut cur_x_1 = v3[0];
        let mut cur_x_2 = v3[0];

        for i in ((v1[1] as i64)..(v3[1] as i64)).rev(){



            for j in (cur_x_1.min(cur_x_2) as i64)..(cur_x_1.max(cur_x_2)as i64) {
                // draw line

                if i<0 || i>=screen_height || j<0 || j>=screen_width {
                    return;
                }
    
                let index = (i*(screen_width) + j) as usize;
                pixel_buffer[index] = self.color;

            }

            cur_x_1 -= inv_slope_1;
            cur_x_2 -= inv_slope_2;

        }
    }



    
}