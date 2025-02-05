
use crate::mesh::mesh::Mesh;
use crate::models::cube::Cube;

pub struct World{
    pub elements: Vec<Box<dyn Mesh>>,
    pub idx_vec_running: Vec<u32>,
}

impl World{

    pub fn new() -> World {


        /*
        ADJUST 'amount_high' and 'amount_wide' for quick changes in the models.
        you can also make a custom shape of any kind if you implement it with 'mesh' traits
         */


        let mut elements: Vec<Box<dyn Mesh>> = Vec::new();

        // //test with a row of cubes
        let side_length = 3.;
        let amount_wide = 20;
        let amount_high = 20;
        let mut count: u32 = 0;
        for i in -amount_wide..amount_wide{
            for j in -amount_high..amount_high{
                count +=1;
                    let idx = i as f32;
                    let jdx = j as f32;
                    let cube = Cube::new(side_length * idx,side_length * jdx,0.,side_length, 0xFFFFFF * (count %2));
                    elements.push(Box::new(cube));
            }
        }

        
        // let teapot = Imported::new("../../3d_models/teapot.obj",100.,0.,0.,0.);

        // elements.push(Box::new(teapot));
        let g: f32 = 1.22/2.;
        let x_pt = g.tan();


        let cube = Cube::new(x_pt,0.,-1.,0.5, 0xFFFFFF);
        elements.push(Box::new(cube));

        let cube = Cube::new(-x_pt,0.,-1.,0.5, 0xFFFFFF);

        elements.push(Box::new(cube));

        // let cube = Cube::new(-40.,-20.,3.,40., 0xFFFFFF);
        // elements.push(Box::new(cube));
        
        // let cube = Cube::new(20.,-20.,3.,40., 0xFFFFFF);
        // elements.push(Box::new(cube));
        
        let mut running_total = 0;

        let mut idx_vec_running: Vec<u32> = Vec::new();    

        for mesh in &elements {
            idx_vec_running.push(running_total);
            running_total += mesh.num_vertices() as u32;
        }



        
        World {
            elements, idx_vec_running,
        }
    }

}