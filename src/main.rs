use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;

use rok::*;
use rustyline::error::ReadlineError;
use rustyline::DefaultEditor;

fn main() {
  env_logger::init();
  println!("rok {}", env!("CARGO_PKG_VERSION"));

  let data_dir = match directories::ProjectDirs::from("github", "AshyIsMe", "rok") {
    Some(dirs) => dirs.data_dir().to_path_buf(),
    None => PathBuf::new(),
  };
  fs::create_dir_all(&data_dir).unwrap();
  let hf: PathBuf = data_dir.join("khistory");

  let mut rl = DefaultEditor::new().unwrap();
  if hf.exists() {
    rl.load_history(&hf).unwrap();
  }

  let mut env = Env { names: HashMap::new(), parent: None };

  loop {
    let readline = rl.readline(" ");
    match readline {
      Ok(line) => {
        let _ = rl.add_history_entry(&line);
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
          let r = eval(&mut env, scan(&line).unwrap());
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
  // println!("rl.save_history({:?})", hf);
  let _ = rl.save_history(&hf);
}
