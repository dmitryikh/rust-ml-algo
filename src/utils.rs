use std::fs::File;
use std::io;
use std::str;
use std::io::prelude::*;

pub fn read_csv_job<S, F>( fname: &str,
                           csv_skip_header: u32,
                           sep: char,
                           usecols_opt: Option<&[usize]>,
                           skip_errors: bool,
                           mut job: F,
                      ) -> Result<u32, String>
    where S : Clone + Default + str::FromStr,
          F : FnMut(&Vec<S>)
{
    let mut usecols: Vec<usize>;
    let mut max_col: usize;

    match usecols_opt {
        Some(ref vec) if vec.len() == 0  => { return Err("usecols_opt is zero".to_string()); },
        Some(vec) => { 
            usecols = vec.to_vec();
            max_col = vec.iter().max().unwrap() + 1;
        },
        None => {
            usecols = Vec::new();
            max_col = 0;
        },
    };
    let file = File::open(fname).map_err(|_| format!("Can't open file {}", fname))?; 
    let mut buf_reader = io::BufReader::new(file);
    let mut line = String::new();
    let mut no = 0;
    let mut counter: u32 = 0;
    let mut vals: Vec<S> = vec![S::default(); usecols.len()];
    for _ in 0..csv_skip_header {
        line.clear();
        buf_reader.read_line(&mut line).map_err(|_| format!("Can't read line {}", no))?;
        no += 1;
    }
    line.clear();
    'outer: loop {
        line.clear();
        match buf_reader.read_line(&mut line) {
            Err(_) => { break 'outer; }
            Ok(_) => {
                let line2 = line.trim();
                no += 1;
                if line2.len() == 0 { break; }
                let tokens: Vec<&str> = line2.split(sep).collect();
                if usecols.len() == 0 {
                    usecols = (0..tokens.len()).collect();
                    max_col = tokens.len();
                    vals = vec![S::default(); usecols.len()];
                }
                if tokens.len() < max_col {
                    return Err(format!("line {}: found {} columns, expecting at least {}",
                                                       no, tokens.len(), max_col));
                }
                for (i, index) in usecols.iter().enumerate() {
                    vals[i] = match tokens[*index].parse() {
                        Ok(val) => { val },
                        Err(_) if skip_errors => { continue 'outer; },
                        Err(_) => {
                            return Err(format!("line {}: can't convert \"{}\" into float", no, tokens[*index]));
                        }
                    };
                }
                job(&vals);
                counter += 1;
            }
        }
    }
    Ok(counter)
}

#[test]
fn read_csv() {
    let mut counter = 0;
    read_csv_job::<f32, _>( "data/test.csv",
                  1,
                  ',',
                  None,
                  false, 
                  |vals| {
        match counter {
            0 => { assert_eq!(*vals, [1.0,2.0,3.0]); },
            1 => { assert_eq!(*vals, [4.5,1.4,999.9]); },
            _ => { },
        };
        counter += 1;
        println!("vals: {:?}", *vals);
    },
                ).unwrap();
}


#[test]
#[should_panic(expected = "Can\\'t open file")]
fn noexist_csv() {
    read_csv_job::<f32, _>( "data/non_sense_file.csv",
                  1,
                  ',',
                  None,
                  false, 
                  |_| { }
                ).unwrap();
}
