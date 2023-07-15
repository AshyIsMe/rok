use std::io::{self, Write};
use rok::*;

fn main() {
    println!("rok v0.000");

    let mut buffer = String::new();
    let stdin = io::stdin();
    let mut stdout = io::stdout();
    loop {
        stdout.write_all(b" ").unwrap(); //prompt
        stdout.flush().unwrap();
        stdin.read_line(&mut buffer).unwrap();

        let r = eval(scan(&buffer).unwrap());
        // stdout.write_all(b"{r:?}\n").unwrap();
        println!("{r:?}\n");

        buffer.truncate(0);
    }
}
