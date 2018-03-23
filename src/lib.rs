extern crate rand;
extern crate permutohedron;

pub mod utils;
pub mod matrix;
pub mod kmeans;
pub mod em;

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}
