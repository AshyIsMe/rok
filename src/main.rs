use std::io::{self, Write};

fn main() {
    println!("rok v0.000");

    let mut buffer = String::new();
    let stdin = io::stdin();
    let mut stdout = io::stdout();
    loop {
        stdout.write_all(b" ").unwrap(); //prompt
        stdout.flush().unwrap();
        stdin.read_line(&mut buffer).unwrap();

        stdout.write_all(b"TODO: Actual eval!\n").unwrap(); //prompt

        buffer.truncate(0);
    }
}
