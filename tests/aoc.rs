use std::collections::HashMap;

use polars::prelude::*;
use roklang::*;

use roklang::KW::*;

#[test]
fn test_aoc2015_12_01() {
  let mut env = Env { names: HashMap::new(), parent: None };

  assert_eq!(
    eval(&mut env, scan(r#"(+/-1*")"=s)+(+/"("=s:"()")"#).unwrap()).unwrap(),
    Noun(K::Int(Some(0)))
  );

  // TODO
  // assert_eq!(
  //   eval(&mut env, scan(r#"+//1 -1*"()"=\:s:"((()))""#).unwrap()).unwrap(),
  //   Noun(K::Int(Some(0)))
  // );
}

#[test]
fn test_aoc2015_12_01_p2() {
  let mut env = Env { names: HashMap::new(), parent: None };

  assert_eq!(
    eval(&mut env, scan(r#"1+*&-1=+\+/1 -1*"()"=\:s:"(()))()")"#).unwrap()).unwrap(),
    Noun(K::Int(Some(5)))
  );
}
