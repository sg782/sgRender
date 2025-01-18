use crate::mesh::face::Face;
use crate::mesh::point::{self, Point};
use crate::mesh::mesh::Mesh;

use std::fs::File;
use std::io::{self, BufRead};
use std::path::Path;

// https://paulbourke.net/dataformats/obj/

// for now, only parse v and f values

pub struct Imported {
    vertices: Vec<Point>,
    faces: Vec<Face>,
}

impl Mesh for Imported {
    fn vertices(&self) -> &Vec<Point> {
        &self.vertices
    }

    fn faces(&self) -> &Vec<Face> {
        &self.faces
    }
}

impl Imported {
    pub fn new(file_path: &str, scale: f64) -> Imported{

        if !file_path.ends_with(".obj"){
            panic!(".obj filetype required!!");
        }

        let mut faces: Vec<Face> = Vec::new();
        let mut vertices: Vec<Point> = Vec::new();
        if let Ok(lines) = Imported::read_lines(file_path) {

            for line in lines.map_while(Result::ok) {
                if line.starts_with("v "){
                    // if a vertex
                    
                    let mut point_data: Vec<f64> = Vec::new();
                    let itr = line[2..line.len()].split(" ");
                    
                    for coord in itr {
                        if let Ok(num) = coord.parse::<f64>() {
                            point_data.push(scale*num);
                        }else{
                            panic!("Non numerical data: {}",coord);
                        }
                    }
                    vertices.push(Point::new_from_vec(point_data));

                }   

                if line.starts_with("f "){
                    /*
                        https://paulbourke.net/dataformats/obj/

                        face data is written like

                        1//3 5//6 3//4

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

                    println!("{:?}", face_data);

                    if(face_data.len()!=3){
                        continue;
                    }

                    faces.push(Face::new_from_vec(face_data));

                }
            }
        }else{
            panic!("File '{}' could not be read!",file_path);
        }

        /*
        read and parse file

        only care about things marked as 'v' and 'f' for now        
        
         */

        Imported {
            vertices,
            faces
        }
    }

    // straight from the rust docs
    fn read_lines<P>(filename: P) -> io::Result<io::Lines<io::BufReader<File>>>
    where P: AsRef<Path>, {
        let file = File::open(filename)?;
        Ok(io::BufReader::new(file).lines())
    }
}