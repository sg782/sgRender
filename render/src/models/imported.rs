use crate::mesh::face::Face;
use crate::mesh::point::Point;
use crate::mesh::mesh::Mesh;

use std::f32::INFINITY;
use std::fs::File;
use std::io::{self, BufRead};
use std::path::Path;
use nalgebra::{Vector2,Vector3};


// https://paulbourke.net/dataformats/obj/

// for now, only parse v and f values

pub struct Imported {
    vertices: Vec<Point>,
    faces: Vec<Face>,
    bounding_box: Vector2<Vector3<f32>>,
    num_vertices: usize, 
}

impl Mesh for Imported {
    fn vertices(&self) -> &Vec<Point> {
        &self.vertices
    }

    fn faces(&self) -> &Vec<Face> {
        &self.faces
    }

    fn bounding_box(&self) -> &Vector2<Vector3<f32>> {
       &self.bounding_box
    }

    fn color(&self) -> u32 {
        // default color for now
        0xFF00FF
    }
    fn num_vertices(&self) -> usize {
        self.num_vertices
    }
}

impl Imported {
    pub fn new(file_path: &str, scale: f32,offset_x: f32, offset_y: f32, offset_z: f32) -> Imported{

        let mut faces: Vec<Face> = Vec::new();
        let mut vertices: Vec<Point> = Vec::new();

        // corners for bounding cube, will iteratively find the actual points
        let min_bounding_vertex: Vector3<f32> = Vector3::new(INFINITY,INFINITY,INFINITY);
        let max_bounding_vertex: Vector3<f32> = Vector3::new(-INFINITY,-INFINITY,-INFINITY);
        let mut bounding_box: Vector2<Vector3<f32>> = Vector2::new(min_bounding_vertex,max_bounding_vertex);


        Imported::fill_model_data(file_path, scale, offset_x, offset_y, offset_z, &mut faces, &mut vertices,& mut bounding_box);

        let num_vertices = vertices.len();

        Imported {
            vertices,
            faces,
            bounding_box,
            num_vertices,
        }
    }

    // straight from the rust docs
    fn read_lines<P>(filename: P) -> io::Result<io::Lines<io::BufReader<File>>>
    where P: AsRef<Path>, {
        let file = File::open(filename)?;
        Ok(io::BufReader::new(file).lines())
    }

    fn fill_model_data(file_path: &str, scale: f32,offset_x: f32, offset_y: f32, offset_z: f32, faces: & mut Vec<Face>, vertices: &mut Vec<Point>,bounding_box: & mut Vector2<Vector3<f32>>){
        if !file_path.ends_with(".obj"){
            panic!(".obj filetype required!!");
        }

        if let Ok(lines) = Imported::read_lines(file_path) {

            for line in lines.map_while(Result::ok) {
                if line.starts_with("v "){
                    // if a vertex
                    
                    let mut point_data: Vec<f32> = Vec::new();
                    let itr = line[2..line.len()].split(" ");
                    
                    for coord in itr {
                        if let Ok(num) = coord.parse::<f32>() {
                            point_data.push(scale*num);
                        }else{
                            panic!("Non numerical data: {}",coord);
                        }
                    }

                    assert_eq!(point_data.len(),3);
                    point_data[0] += offset_x;
                    point_data[1] += offset_y;
                    point_data[2] += offset_z;

                    // update cube bounds
                    bounding_box[0][0] = bounding_box[0][0].min(point_data[0]);
                    bounding_box[0][1] = bounding_box[0][1].min(point_data[1]);
                    bounding_box[0][2] = bounding_box[0][2].min(point_data[2]);

                    bounding_box[1][0] = bounding_box[1][0].max(point_data[0]);
                    bounding_box[1][1] = bounding_box[1][1].max(point_data[1]);
                    bounding_box[1][2] = bounding_box[1][2].max(point_data[2]);



                    vertices.push(Point::new_from_vec(point_data));

                }   

                if line.starts_with("f "){
                    /*
                        https://paulbourke.net/dataformats/obj/

                        face data is written like 1//3 5//6 3//4
                        where the first value is the vertex id, and the second is the 'vertex normal' id. I am currently ignoring vn's,
                        so we will just read the first integer of each block
                        it can also be written with a third value in each block, but we ignore that too for now
                     */


                    let mut face_data: Vec<i64> = Vec::new();
                    let itr = line[2..line.len()].split(" ");

                    
                    for block in itr {

                        if block.contains("//"){
                            let face_vertex: Vec<&str> = block.split("//").collect();
                            
                            if let Ok(num) = face_vertex[0].parse::<i64>() {
                                face_data.push(num-1); //0 indexed
                            }else{
                                panic!("Non numerical data: {}",face_vertex[0]);
                            }
                        }else{
                            let face_vertex: Vec<&str> = block.split("/").collect();

                            if let Ok(num) = face_vertex[0].parse::<i64>() {
                                face_data.push(num-1); //0 indexed
                            }else{
                                panic!("Non numerical data: {}",face_vertex[0]);
                            }
                        }

                        
                    }

                    if face_data.len()!=3 {
                        continue;
                    }

                    faces.push(Face::new_from_vec(face_data));

                }
            }
        }else{
            panic!("File '{}' could not be read!",file_path);
        }

    }
}