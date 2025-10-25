use roklang::*;
use std::collections::HashMap;
use std::fs::{File};
use std::io::Read;

fn k_eval(s: &str) -> Result<KW, &str> {
  let mut env = Env { names: HashMap::new(), parent: None };

  // let r = eval(&mut env, scan(s).unwrap()).unwrap().unwrap_noun();
  match eval(&mut env, scan(s).unwrap()) {
    Ok(r) => {
      // let r = r.unwrap_noun();
      // println!("k_eval({}) = {}", s, r);
      Ok(r)
    }
    Err(e) => {
      println!("k_eval({}) Err: {}", s, e);
      Err(e)
    }
  }
}
// fn k_evals(s: &str) -> String { format!("{}", k_eval(s)) }

#[test]
fn test_ngnk_tests() {
  let mut file = match File::open("ngn/k/t/t.k") {
    Ok(file) => file,
    // Err(_) => panic!("no such file"),
    Err(_) => {
      println!("Skipping test_ngnk_tests() t.k does not exist. Run bench-vs-ngn-k.sh first.");
      return;
    }
  };
  let mut file_contents = String::new();
  file.read_to_string(&mut file_contents).ok().expect("failed to read!");
  let lines: Vec<String> = file_contents.split("\n").map(|s: &str| s.to_string()).collect();

  assert!(lines.len() > 0);
  let mut test_count = 0;
  let mut failed_tests = 0;

  // TODO add support for these lines
  let skiplines = [9, 10, 14, 15, 28, 30, 31, 32, 34];

  for (i, l) in lines.iter().enumerate() {
    if skiplines.contains(&i) {
      test_count += 1;
      failed_tests += 1;
      println!("\nskipping line {} known failure: {}", i + 1, l);
    } else {
      println!("\nline {}: {}", i + 1, l);
      let t: Vec<&str> = l.split(" / ").collect();
      if t.len() != 2 {
        println!("Skipping dud line: {}", l);
      } else {
        test_count += 1;
        //   assert_eq!(k_eval(t[0]), k_eval(t[1]));
        let res = k_eval(t[0]);
        if res != k_eval(t[1]) {
          failed_tests += 1;
          println!("Failed test: ({failed_tests}/{test_count}): {}", l);
          match res {
            Ok(k) => println!("{}", k),
            Err(_) => println!("{:?}", res),
          }
        }
      }
    }
  }
  println!("test_count: {}\nfailed_tests: {}", test_count, failed_tests);
  assert!(failed_tests == 0);
}
