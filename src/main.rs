use rok::*;
use rustyline::error::ReadlineError;
use rustyline::DefaultEditor;

fn main() {
    println!("rok {}", env!("CARGO_PKG_VERSION"));

    let mut rl = DefaultEditor::new().unwrap();
    loop {
        let readline = rl.readline(" ");
        match readline {
            Ok(line) => {
                let r = eval(scan(&line).unwrap());
                println!("{r:?}\n");
            },
            Err(ReadlineError::Interrupted) | Err(ReadlineError::Eof) => {
                break
            },
            Err(err) => {
                println!("Error: {:?}", err);
                break
            }
        }
    }
}
