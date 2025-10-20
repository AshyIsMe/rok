use roklang::KW::*;
use roklang::*;
use std::collections::HashMap;
use std::fs::{self, File};
use std::io::Read;

fn k_eval(s: &str) -> K {
  let mut env = Env { names: HashMap::new(), parent: None };

  let r = eval(&mut env, scan(s).unwrap()).unwrap().unwrap_noun();
  println!("k_eval({}) = {}", s, r);
  r
}
fn k_evals(s: &str) -> String { format!("{}", k_eval(s)) }

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

  assert!(lines.len()>0);
  let mut test_count = 0;
  let mut failed_tests = 0;
  for l in lines.iter() {
      println!("line: {}", l);
      let t: Vec<&str> = l.split(" / ").collect();
      if t.len() != 2 {
          println!("Skipping dud line: {}", l);
      } else {
          test_count += 1;
        //   assert_eq!(k_eval(t[0]), k_eval(t[1]));
          if k_eval(t[0]) != k_eval(t[1]) {
              println!("Failed test: {}", l);
              failed_tests += 1;
          }
      }
  }
  println!("test_count: {}\nfailed_tests: {}", test_count, failed_tests);
  assert!(failed_tests == 0);
}
