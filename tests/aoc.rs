use std::collections::HashMap;

use roklang::*;

use roklang::KW::*;

#[test]
fn test_aoc2015_12_01() {
  let mut env = Env { names: HashMap::new(), parent: None };

  assert_eq!(
    eval(&mut env, scan(r#"(+/-1*")"=s)+(+/"("=s:"()")"#).unwrap()).unwrap(),
    Noun(K::Int(Some(0)))
  );
}

#[ignore]
#[test]
fn test_aoc2015_12_01_v2() {
  let mut env = Env { names: HashMap::new(), parent: None };

  // TODO
  assert_eq!(
    eval(&mut env, scan(r#"+//1 -1*"()"=\:s:"((()))""#).unwrap()).unwrap(),
    Noun(K::Int(Some(0)))
  );
}

#[test]
fn test_aoc2015_12_01_p2() {
  let mut env = Env { names: HashMap::new(), parent: None };

  assert_eq!(
    eval(&mut env, scan(r#"1+*&-1 = +\(-1*")"=s)+("("=s:"(()))()")"#).unwrap()).unwrap(),
    Noun(K::Int(Some(5)))
  );
}

#[ignore]
#[test]
fn test_aoc2015_12_01_p2_v2() {
  let mut env = Env { names: HashMap::new(), parent: None };

  // TODO
  assert_eq!(
    eval(&mut env, scan(r#"1+*&-1=+\+/1 -1*"()"=\:s:"(()))()")"#).unwrap()).unwrap(),
    Noun(K::Int(Some(5)))
  );
}

#[test]
fn test_aoc2015_12_02() {
  let mut env = Env { names: HashMap::new(), parent: None };

  // 20x3x11
  // 15x27x5
  // 6x29x7
  // {"x"\x}'s:("20x3x11";"15x27x5";"6x29x7")
  let res =
    eval(&mut env, scan(r#".''{"x"\x}'s:("20x3x11";"15x27x5";"6x29x7")"#).unwrap()).unwrap();
  let expected = eval(&mut env, scan(r#"((20 3 11);(15 27 5);(6 29 7))"#).unwrap()).unwrap();
  assert_eq!(res, expected);

  let res = eval(
    &mut env,
    scan(r#"{((x@0)*(x@1);(x@1)*(x@2);(x@0)*(x@2))}'{.'"x"\x}'s:("20x3x11";"15x27x5";"6x29x7")"#)
      .unwrap(),
  )
  .unwrap();
  let expected =
    eval(&mut env, scan(r#"((60 33 220);(405 135 75);(174 203 42))"#).unwrap()).unwrap();
  assert_eq!(res, expected);

  let s = r#"{l:x@0;w:x@1;h:x@2;d:(l*w;w*h;l*h);+/(2*d),&/d}' {.'"x"\x}'s:("2x3x4";"1x1x10")"#;
  let res = eval(&mut env, scan(s).unwrap()).unwrap();
  let expected = eval(&mut env, scan("58 43").unwrap()).unwrap();
  assert_eq!(res, expected);

  // TODO Why is this so slow?
  let s = r#"+/{l:x@0;w:x@1;h:x@2;d:(l*w;w*h;l*h);+/(2*d),&/d}' {.'"x"\x}'s:0:"tests/aoc/2015/day02.txt""#;
  let res = eval(&mut env, scan(s).unwrap()).unwrap();
  let expected = eval(&mut env, scan("1606483").unwrap()).unwrap();
  assert_eq!(res, expected);
}

#[ignore]
#[test]
fn test_aoc2015_12_02_v2() {
  let mut env = Env { names: HashMap::new(), parent: None };

  let s = r#"(2*+//s)+/&/s:(*|x)*':x:`I$+"x"\'0:"tests/aoc/2015/day02.txt""#;
  let res = eval(&mut env, scan(s).unwrap()).unwrap();
  let expected = eval(&mut env, scan("1606483").unwrap()).unwrap();
  assert_eq!(res, expected);
}

#[test]
fn test_aoc2015_12_02_p2() {
  let mut env = Env { names: HashMap::new(), parent: None };

  let s = r#"{(*/x)+/2*2#x[<x]}'(2 3 4; 1 1 10)"#;
  let res = eval(&mut env, scan(s).unwrap()).unwrap();
  let expected = eval(&mut env, scan("34 14").unwrap()).unwrap();
  assert_eq!(res, expected);
}
