use itertools::Itertools;
use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;
use std::time::SystemTime;

use roklang::*;
use rustyline::error::ReadlineError;
use rustyline::DefaultEditor;

fn help() -> &'static str {
  // TODO better help, up to date with nyi list

  "verbs:
: set return
+ plus flip
- negate minus
* first times
% sqrt divide
! iota|odometer|keys dict|mod
& where min
| reverse max
< asc lesser
> desc greater
= imat|group equal
~ not match
, enlist concat
^ isnull except|fill
# count take|reshape
_ floor|lowercase drop|delete|cut
$ string pad|cast              $[c;t;f] cond
? unique|randfloat find|random splice
@ type at
. eval|values dot
"
}

fn main() {
  env_logger::init();
  println!("rok {}, type \\ for more info", env!("CARGO_PKG_VERSION"));

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
          println!("{}", help());
          println!("adverbs: {}", adverbs_table().keys().join(" "));
        } else if line.trim_end() == "\\\\" {
          //quit
          break;
        } else if line.starts_with("\\s ") {
          //scan tokens instead of eval
          let r = scan(&line[2..]).unwrap();
          println!("{r:?}");
        } else if line.starts_with("\\t ") {
          //time eval
          let now = SystemTime::now();
          let r = eval(&mut env, scan(&line[2..]).unwrap());
          match r {
            Ok(_kw) => println!("{}", now.elapsed().unwrap().as_millis()),
            _ => println!("{:?}", r),
          }
        } else if line.starts_with("\\d ") {
          //debug print
          println!("debug print:");
          let r = eval(&mut env, scan(&line[2..]).unwrap());
          println!("{r:?}");
        } else {
          let r = eval(&mut env, scan(&line).unwrap());
          match r {
            Ok(kw) => println!("{kw}"),
            _ => println!("{:?}", r),
          }
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
