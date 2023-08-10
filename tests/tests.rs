use polars::prelude::*;
use rok::*;

use rok::KW::*;

#[test]
fn test_scan() {
  assert_eq!(scan("1").unwrap(), vec![Noun(K::Bool(1u8))]);
  assert_eq!(scan("2").unwrap(), vec![Noun(K::Int(Some(2)))]);
  assert_eq!(scan("3.14").unwrap(), vec![Noun(K::Float(3.14))]);
  assert_eq!(scan("1 0 1 0 1").unwrap(), vec![Noun(K::BoolArray(arr!([1, 0, 1, 0, 1u8])))]);
  assert_eq!(scan("1 2 3").unwrap(), vec![Noun(K::IntArray(arr!([1, 2, 3i64])))]);
  assert_eq!(scan("1 2 3.14").unwrap(), vec![Noun(K::FloatArray(arr!([1., 2., 3.14])))]);
}

#[test]
fn test_eval() {
  assert_eq!(
    scan("2 + 2").unwrap(),
    vec![Noun(K::Int(Some(2))), Verb { name: "+".to_string() }, Noun(K::Int(Some(2)))]
  );
  assert_eq!(eval(scan("2 + 2").unwrap()).unwrap(), Noun(K::Int(Some(4))));
  assert_eq!(eval(scan("2 + 2 + 2").unwrap()).unwrap(), Noun(K::Int(Some(6))));

  assert_eq!(
    scan("1.0 + 2.0").unwrap(),
    vec![Noun(K::Float(1.0)), Verb { name: "+".to_string() }, Noun(K::Float(2.0))]
  );
  assert_eq!(eval(scan("1.0 + 2.0").unwrap()).unwrap(), Noun(K::Float(3.0)));
  assert_eq!(eval(scan("1.0 - 2.0").unwrap()).unwrap(), Noun(K::Float(-1.0)));
  assert_eq!(eval(scan("2.0 * 2.0").unwrap()).unwrap(), Noun(K::Float(4.0)));
  assert_eq!(eval(scan("10 % 2").unwrap()).unwrap(), Noun(K::Int(Some(5))));

  assert_eq!(
    eval(scan("1 0 1 0 1 + 2").unwrap()).unwrap(),
    Noun(K::IntArray(arr!([3, 2, 3, 2, 3])))
  );

  // assert_eq!(eval(scan("1 2 3 + 4 5 6").unwrap()).unwrap(), Noun(K::IntArray(arr!([5, 7, 9]))));
}

#[test]
fn test_parens() {
  assert_eq!(scan("(1)").unwrap(), vec![KW::LP, Noun(K::Bool(1)), KW::RP]);
  assert_eq!(eval(scan("(1)").unwrap()).unwrap(), Noun(K::Bool(1)));
  assert_eq!(eval(scan("(1+1)").unwrap()).unwrap(), Noun(K::Int(Some(2))));
  assert_eq!(eval(scan("(1+1)+3").unwrap()).unwrap(), Noun(K::Int(Some(5))));
  assert_eq!(eval(scan("(1*1)+(3*3)").unwrap()).unwrap(), Noun(K::Int(Some(10))));
  assert_eq!(
    eval(scan("((2 + 0 + 0 + 1 - 1) * (3*1)) + 1").unwrap()).unwrap(),
    Noun(K::Int(Some(7)))
  );
}

#[test]
fn test_array_maths() {
  // TODO: Why do these fail the assert_eq even though they look correctly matched? (pointers vs values maybe?)
  // assert_eq!(eval(scan("3.14 + 1 2 3").unwrap()).unwrap(), Noun(K::FloatArray(arr!([4.14, 5.14, 6.14]))));
  // assert_eq!(eval(scan("3.14 + 1.0 2.0 3.0").unwrap()).unwrap(), Noun(K::FloatArray(arr!([4.14, 5.14, 6.14]))));
  assert_eq!(
    format!("{:?}", eval(scan("3.14 + 1 2 3").unwrap()).unwrap()),
    format!("{:?}", Noun(K::FloatArray(arr!([4.14, 5.14, 6.14]))))
  );
  assert_eq!(
    format!("{:?}", eval(scan("3.14 + 1.0 2.0 3.0").unwrap()).unwrap()),
    format!("{:?}", Noun(K::FloatArray(arr!([4.14, 5.14, 6.14]))))
  );

  assert_eq!(
    format!("{:?}", eval(scan("1.1 2.1 + 1").unwrap()).unwrap()),
    format!("{:?}", Noun(K::FloatArray(arr!([2.1, 3.1]))))
  );
  assert_eq!(
    format!("{:?}", eval(scan("1.1 2.1 + 2").unwrap()).unwrap()),
    format!("{:?}", Noun(K::FloatArray(arr!([3.1, 4.1]))))
  );
  assert_eq!(
    format!("{:?}", eval(scan("1.1 2.1 + 2.0").unwrap()).unwrap()),
    format!("{:?}", Noun(K::FloatArray(arr!([3.1, 4.1]))))
  );
  assert_eq!(
    format!("{:?}", eval(scan("1.1 2.1 + 1 1").unwrap()).unwrap()),
    format!("{:?}", Noun(K::FloatArray(arr!([2.1, 3.1]))))
  );
  assert_eq!(
    format!("{:?}", eval(scan("1.1 2.1 + 2 2").unwrap()).unwrap()),
    format!("{:?}", Noun(K::FloatArray(arr!([3.1, 4.1]))))
  );
  assert_eq!(
    format!("{:?}", eval(scan("1.1 2.1 + 2.0 2.0").unwrap()).unwrap()),
    format!("{:?}", Noun(K::FloatArray(arr!([3.1, 4.1]))))
  );

  assert_eq!(
    format!("{:?}", eval(scan("1.1 2.1 - 1").unwrap()).unwrap()),
    format!("{:?}", Noun(K::FloatArray(arr!([0.1, 1.1]))))
  );
  assert_eq!(
    format!("{:?}", eval(scan("1.1 2.1 - 2").unwrap()).unwrap()),
    format!("{:?}", Noun(K::FloatArray(arr!([-0.9, 0.1]))))
  );
  assert_eq!(
    format!("{:?}", eval(scan("1.1 2.1 - 2.0").unwrap()).unwrap()),
    format!("{:?}", Noun(K::FloatArray(arr!([-0.9, 0.1]))))
  );
  assert_eq!(
    format!("{:?}", eval(scan("1.1 2.1 - 1 1").unwrap()).unwrap()),
    format!("{:?}", Noun(K::FloatArray(arr!([0.1, 1.1]))))
  );
  assert_eq!(
    format!("{:?}", eval(scan("1.1 2.1 - 2 2").unwrap()).unwrap()),
    format!("{:?}", Noun(K::FloatArray(arr!([-0.9, 0.1]))))
  );
  assert_eq!(
    format!("{:?}", eval(scan("1.1 2.1 - 2.0 2.0").unwrap()).unwrap()),
    format!("{:?}", Noun(K::FloatArray(arr!([-0.9, 0.1]))))
  );

  assert_eq!(
    format!("{:?}", eval(scan("1.1 2.1 * 1").unwrap()).unwrap()),
    format!("{:?}", Noun(K::FloatArray(arr!([1.1, 2.1]))))
  );
  assert_eq!(
    format!("{:?}", eval(scan("1.1 2.1 * 2").unwrap()).unwrap()),
    format!("{:?}", Noun(K::FloatArray(arr!([2.2, 4.2]))))
  );
  assert_eq!(
    format!("{:?}", eval(scan("1.1 2.1 * 2.0").unwrap()).unwrap()),
    format!("{:?}", Noun(K::FloatArray(arr!([2.2, 4.2]))))
  );
  assert_eq!(
    format!("{:?}", eval(scan("1.1 2.1 * 1 1").unwrap()).unwrap()),
    format!("{:?}", Noun(K::FloatArray(arr!([1.1, 2.1]))))
  );
  assert_eq!(
    format!("{:?}", eval(scan("1.1 2.1 * 2 2").unwrap()).unwrap()),
    format!("{:?}", Noun(K::FloatArray(arr!([2.2, 4.2]))))
  );
  assert_eq!(
    format!("{:?}", eval(scan("1.1 2.1 * 2.0 2.0").unwrap()).unwrap()),
    format!("{:?}", Noun(K::FloatArray(arr!([2.2, 4.2]))))
  );

  assert_eq!(
    format!("{:?}", eval(scan("1.1 2.1 % 1").unwrap()).unwrap()),
    format!("{:?}", Noun(K::FloatArray(arr!([1.1, 2.1]))))
  );
  assert_eq!(
    format!("{:?}", eval(scan("1.1 2.1 % 2").unwrap()).unwrap()),
    format!("{:?}", Noun(K::FloatArray(arr!([0.55, 1.05]))))
  );
  assert_eq!(
    format!("{:?}", eval(scan("1.1 2.1 % 2.0").unwrap()).unwrap()),
    format!("{:?}", Noun(K::FloatArray(arr!([0.55, 1.05]))))
  );
  assert_eq!(
    format!("{:?}", eval(scan("1.1 2.1 % 1 1").unwrap()).unwrap()),
    format!("{:?}", Noun(K::FloatArray(arr!([1.1, 2.1]))))
  );
  assert_eq!(
    format!("{:?}", eval(scan("1.1 2.1 % 2 2").unwrap()).unwrap()),
    format!("{:?}", Noun(K::FloatArray(arr!([0.55, 1.05]))))
  );
  assert_eq!(
    format!("{:?}", eval(scan("1.1 2.1 % 2.0 2.0").unwrap()).unwrap()),
    format!("{:?}", Noun(K::FloatArray(arr!([0.55, 1.05]))))
  );
}

#[test]
fn test_iota() {
  assert_eq!(
    format!("{:?}", eval(scan("! 4").unwrap()).unwrap()),
    format!("{:?}", Noun(K::IntArray(arr!([0, 1, 2, 3i64]))))
  );
}

#[test]
fn test_strings() {
  assert_eq!(
    format!("{:?}", eval(scan("\"a\"").unwrap()).unwrap()),
    format!("{:?}", Noun(K::Char('a')))
  );
  assert_eq!(
    format!("{:?}", eval(scan("\"\"").unwrap()).unwrap()),
    format!("{:?}", Noun(K::CharArray(Series::new("", ""))))
  );
  assert_eq!(
    format!("{:?}", eval(scan("\"abcABC\"").unwrap()).unwrap()),
    format!("{:?}", Noun(K::CharArray(Series::new("", "abcABC"))))
  );
}

#[test]
fn test_symbols() {
  assert_eq!(
    format!("{:?}", eval(scan("`a").unwrap()).unwrap()),
    format!("{:?}", Noun(K::Symbol("a".to_string())))
  );
  assert_eq!(
    format!("{:?}", eval(scan("`abc ").unwrap()).unwrap()),
    format!("{:?}", Noun(K::Symbol("abc".to_string())))
  );
//   assert_eq!(
//     format!("{:?}", eval(scan("`a`b`c").unwrap()).unwrap()),
//     format!("{:?}", Noun(K::SymbolArray(Series::new("", whatever-goes-here))))
//   );
}


#[test]
fn test_length_errors() {
  assert_eq!(eval(scan("1 2 3 + 4 5").unwrap()), Err::<KW, &'static str>("length"));
  assert_eq!(eval(scan("1 2 + 3 4 5").unwrap()), Err::<KW, &'static str>("length"));
  assert_eq!(eval(scan("1.0 2.0 + 3 4 5").unwrap()), Err::<KW, &'static str>("length"));
  assert_eq!(eval(scan("1 2 3 - 4 5").unwrap()), Err::<KW, &'static str>("length"));
  assert_eq!(eval(scan("1 2 - 3 4 5").unwrap()), Err::<KW, &'static str>("length"));
  assert_eq!(eval(scan("1.0 2.0 - 3 4 5").unwrap()), Err::<KW, &'static str>("length"));
  assert_eq!(eval(scan("1 2 3 * 4 5").unwrap()), Err::<KW, &'static str>("length"));
  assert_eq!(eval(scan("1 2 * 3 4 5").unwrap()), Err::<KW, &'static str>("length"));
  assert_eq!(eval(scan("1.0 2.0 * 3 4 5").unwrap()), Err::<KW, &'static str>("length"));
  assert_eq!(eval(scan("1 2 3 % 4 5").unwrap()), Err::<KW, &'static str>("length"));
  assert_eq!(eval(scan("1 2 % 3 4 5").unwrap()), Err::<KW, &'static str>("length"));
  assert_eq!(eval(scan("1.0 2.0 % 3 4 5").unwrap()), Err::<KW, &'static str>("length"));
}
