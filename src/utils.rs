use std::fs::File;
use std::io;
use std::str;
use std::fmt;
use std::io::prelude::*;
use std::io::BufWriter;
use std::ops::{Add, Sub, Mul};
use rand::{Rng, Isaac64Rng, SeedableRng};
use rand;

use permutohedron::Heap;

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
                    vals[i] = match tokens[*index].trim().parse() {
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

pub fn write_csv_col<S>( fname: &str,
                         data: &[S],
                         header: Option<&str>,
                       ) -> Result<(), String>
    where S : fmt::Display
{
    let f = File::create(fname).map_err(|_| format!("Can't open file {}", fname))?;
    let mut f = BufWriter::new(f);
    if let Some(ref h) = header {
        write!(f, "{}\n", h).map_err(|_| "Can't write header".to_string())?;
    }
    for v in data {
        write!(f, "{}\n", v).map_err(|_| "Can't write row".to_string())?;
    }
    Ok(())
}

pub fn accuracy(real: &[u32], pred: &[u32]) -> f64 {
    let matches = real.iter().zip(pred.iter()).fold(0, |matches, v| matches + if *v.0 == *v.1 { 1 } else { 0 });
    return if real.is_empty() { 0.0 } else { matches as f64 / real.len() as f64};
}

pub fn accuracy_perm(real: &[u32], pred: &[u32], labels: &[u32]) -> f64 {
    if real.is_empty() || pred.is_empty() { return 0.0; }
    let mut labels = labels.to_vec();
    let heap_algo = Heap::new(&mut labels);
    let mut accuracy_max = 0.0;
    for data in heap_algo {
        let matches = real.iter().zip(pred.iter()).fold(0, |matches, v| matches + if *v.0 == data[*v.1 as usize] { 1 } else { 0 });
        let accuracy = matches as f64 / real.len() as f64;
        if accuracy > accuracy_max { accuracy_max = accuracy; }
    }
    accuracy_max
}

pub fn vec_dot<S>(a: &[S], b: &[S]) -> S
    where S: Default + Copy + Add<Output = S> + Mul<Output = S> {
    a.iter()
        .zip(b.iter())
        .fold(S::default(), |sum, (ai, bi)| sum + *ai * *bi)
}

pub fn vec_sub<S>(a: &[S], b: &[S]) -> Vec<S>
    where S: Default + Copy + Sub<Output = S> {
    a.iter()
        .zip(b.iter())
        .map(|(ai, bi)| *ai - *bi).collect()
}

pub fn rmse_error(real: &[f64], pred: &[f64]) -> f64 {
    (real.iter().zip(pred.iter())
         .fold(0.0, |sum, (&ri, &pi)| sum + (ri - pi).powi(2))
         / real.len() as f64
    ).sqrt()
}

pub fn mae_error(real: &[f64], pred: &[f64]) -> f64 {
    real.iter().zip(pred.iter())
       .fold(0.0, |sum, (&ri, &pi)| sum + (ri - pi).abs())
       / real.len() as f64
}

pub fn isaac_rng(seed: u64) -> Isaac64Rng {
    let seed = if seed == 0 {
        rand::thread_rng().next_u64()
    } else {
        seed
    };
    Isaac64Rng::from_seed(&[seed])
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

#[test]
fn accuracy_cases() {
    let vec1 = [0, 1, 0, 1, 1, 0];
    let vec2 = [0, 1, 1, 1, 0, 1];
    assert_eq!(1.0, accuracy(&vec1, &vec1));
    assert_eq!(0.0, accuracy(&[], &[]));
    assert_eq!(0.5, accuracy(&vec1, &vec2));
}

#[test]
fn accuracy_perm_cases() {
    let vec1 = [0, 1, 0, 1, 1, 0];
    let vec1_mislabeled = [1, 0, 1, 0, 0, 1];
    let vec2 = [0, 1, 1, 1, 0, 1];
    assert_eq!(1.0, accuracy_perm(&vec1, &vec1, &[0, 1]));
    assert_eq!(1.0, accuracy_perm(&vec1, &vec1_mislabeled, &[0, 1]));
    assert_eq!(0.0, accuracy_perm(&[], &[], &[0, 1]));
    assert_eq!(0.5, accuracy_perm(&vec1, &vec2, &[0, 1]));
    assert_eq!(0.5, accuracy_perm(&vec1_mislabeled, &vec2, &[0, 1]));
}

#[test]
fn vec_dot_cases() {
    assert_eq!(vec_dot(&[0, 1, 2], &[4, 5, 6]), 17);
    assert_eq!(vec_dot(&[0, 1, 2], &[4, 5]), 5);
    assert_eq!(vec_dot(&[1.2, 4.5, 0.0], &[0.0, 2.0, 9.6]), 9.0);
}

#[test]
fn vec_sub_cases() {
    assert_eq!(vec_sub(&[0, 1, 2], &[4, 5, 6]), [-4, -4, -4]);
    assert_eq!(vec_sub(&[0, 1, 2], &[4, 5]), [-4, -4]);
    assert_eq!(vec_sub(&[1.2, 4.5, 0.0], &[0.0, 2.0, 9.6]), [1.2, 2.5, -9.6]);
}

#[test]
fn rmse_cases() {
    assert_eq!(rmse_error(&[0.0, 1.0, 0.0], &[0.0, 0.0, 0.0]), (1.0 as f64/ 3.0).sqrt());
    assert_eq!(rmse_error(&[1.0, 2.0, 10.0], &[1.0, 2.0, 10.0]), 0.0);
}

#[test]
fn mae_cases() {
    assert_eq!(mae_error(&[0.0, 1.0, 0.0], &[0.0, 0.0, 0.0]), 1.0 / 3.0);
    assert_eq!(mae_error(&[1.0, 2.0, 10.0], &[1.0, 2.0, 10.0]), 0.0);
}
