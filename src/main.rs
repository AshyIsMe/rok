use rok::*;
use rustyline::error::ReadlineError;
use rustyline::DefaultEditor;

fn main() {
  env_logger::init();
  println!("rok {}", env!("CARGO_PKG_VERSION"));

  let mut rl = DefaultEditor::new().unwrap();
  loop {
    let readline = rl.readline(" ");
    match readline {
      Ok(line) => {
        if line.trim_end() == "\\" {
          //help
          println!("'nyi\n");
        } else if line.trim_end() == "\\\\" {
          //quit
          break;
        } else if line.starts_with("\\s ") {
          //scan tokens instead of eval
          let r = scan(&line[2..]).unwrap();
          println!("{r:?}\n");
        } else {
          let r = eval(scan(&line).unwrap());
          println!("{r:?}\n");
        }
      }
      Err(ReadlineError::Interrupted) | Err(ReadlineError::Eof) => break,
      Err(err) => {
        println!("Error: {:?}", err);
        break;
      }
    }
  }
}
