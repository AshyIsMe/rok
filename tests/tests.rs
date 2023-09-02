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
  assert_eq!(
    format!("{:?}", eval(scan("`a`b`c").unwrap()).unwrap()),
    format!(
      "{:?}",
      Noun(K::SymbolArray(
        Series::new("a", ["a", "b", "c"]).cast(&DataType::Categorical(None)).unwrap()
      ))
    )
  );
  assert_eq!(
    format!("{:?}", eval(scan("`a`b`c`").unwrap()).unwrap()),
    format!(
      "{:?}",
      Noun(K::SymbolArray(
        Series::new("a", ["a", "b", "c", ""]).cast(&DataType::Categorical(None)).unwrap()
      ))
    )
  );
  assert_eq!(
    format!("{:?}", eval(scan("`a `b `c").unwrap()).unwrap()),
    format!(
      "{:?}",
      Noun(K::SymbolArray(
        Series::new("a", ["a", "b", "c"]).cast(&DataType::Categorical(None)).unwrap()
      ))
    )
  );
  assert_eq!(
    format!("{:?}", eval(scan("`a ` `b `c").unwrap()).unwrap()),
    format!(
      "{:?}",
      Noun(K::SymbolArray(
        Series::new("a", ["a", "", "b", "c"]).cast(&DataType::Categorical(None)).unwrap()
      ))
    )
  );
  assert!(scan("`a ` abc").is_err());
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

#[test]
fn test_lists() {
  assert_eq!(
    format!("{:?}", eval(scan("(1 2; 3 4)").unwrap()).unwrap()),
    format!(
      "{:?}",
      Noun(K::List(vec![K::IntArray(arr!([1, 2i64])), K::IntArray(arr!([3, 4i64]))]))
    )
  );
  assert_eq!(
    format!("{:?}", eval(scan("(1;\"a\";2)").unwrap()).unwrap()),
    format!("{:?}", Noun(K::List(vec![K::Bool(1u8), K::Char('a'), K::Int(Some(2))])))
  );
  assert_eq!(
    format!("{:?}", eval(scan("(1+1;2+2)").unwrap()).unwrap()),
    format!("{:?}", Noun(K::IntArray(arr!([2, 4i64]))))
  );

  assert_eq!(
    format!("{:?}", eval(scan("(1;2;3)").unwrap()).unwrap()),
    format!("{:?}", Noun(K::IntArray(arr!([1, 2, 3i64]))))
  );
  assert_eq!(
    format!("{:?}", eval(scan("(1;0;1)").unwrap()).unwrap()),
    format!("{:?}", Noun(K::BoolArray(arr!([1, 0, 1u8]))))
  );
  assert_eq!(
    format!("{:?}", eval(scan("(1;0;1.5)").unwrap()).unwrap()),
    format!("{:?}", Noun(K::FloatArray(arr!([1.0, 0.0, 1.5f64]))))
  );
  assert_eq!(
    format!("{:?}", eval(scan("(\"a\";\"b\";\"c\")").unwrap()).unwrap()),
    format!("{:?}", Noun(K::CharArray(Series::new("", "abc"))))
  );
  // TODO SymbolArray promotion
  // assert_eq!(
  //   format!("{:?}", eval(scan("(`a;`b;`c)").unwrap()).unwrap()),
  //   format!(
  //     "{:?}",
  //     Noun(K::SymbolArray(
  //       Series::new("a", ["a", "b", "c"]).cast(&DataType::Categorical(None)).unwrap()
  //     ))
  //   )
  // );
}

#[test]
fn test_dict() {
  let k =
    K::SymbolArray(Series::new("a", ["a", "b", "c"]).cast(&DataType::Categorical(None)).unwrap());
  let v = K::List(vec![K::Bool(1u8), K::Int(Some(42)), K::Float(3.14)]);
  let d1 = v_d_bang(k, v);
  println!("{:?}", d1);

  let k =
    K::SymbolArray(Series::new("a", ["a", "b", "c"]).cast(&DataType::Categorical(None)).unwrap());
  let v = K::List(vec![K::Bool(1u8)]);
  let d2 = v_d_bang(k, v);
  println!("{:?}", d2);

  let k = K::SymbolArray(Series::new("a", ["a", "b"]).cast(&DataType::Categorical(None)).unwrap());
  let v = K::List(vec![K::Char('a'), K::Int(Some(42))]);
  let d1 = v_d_bang(k, v).unwrap();
  assert_eq!(
    format!("{:?}", eval(scan("`a`b!(\"a\";42)").unwrap()).unwrap()),
    format!("{:?}", KW::Noun(d1))
  );

  let k = K::SymbolArray(Series::new("a", ["a", "b"]).cast(&DataType::Categorical(None)).unwrap());
  let v = K::List(vec![K::Int(Some(1)), K::Int(Some(2))]);
  let d1 = v_d_bang(k, v).unwrap();
  assert_eq!(
    format!("{:?}", eval(scan("`a`b!1 2").unwrap()).unwrap()),
    format!("{:?}", KW::Noun(d1))
  );

  let k = K::SymbolArray(Series::new("a", ["a", "b"]).cast(&DataType::Categorical(None)).unwrap());
  let v = K::List(vec![K::Int(Some(1)), K::Int(None)]);
  let d1 = v_d_bang(k, v).unwrap();
  assert_eq!(
    format!("{:?}", eval(scan("`a`b!1 0N").unwrap()).unwrap()),
    format!("{:?}", KW::Noun(d1))
  );

  let k = K::SymbolArray(Series::new("a", ["a", "b"]).cast(&DataType::Categorical(None)).unwrap());
  let v = K::List(vec![K::Float(1.5), K::Float(2.5)]);
  let d1 = v_d_bang(k, v).unwrap();
  assert_eq!(
    format!("{:?}", eval(scan("`a`b!1.5 2.5").unwrap()).unwrap()),
    format!("{:?}", KW::Noun(d1))
  );

  let k = K::SymbolArray(Series::new("a", ["a", "b"]).cast(&DataType::Categorical(None)).unwrap());
  let v = K::List(vec![K::Char('a'), K::Char('b')]);
  let d1 = v_d_bang(k, v).unwrap();
  assert_eq!(
    format!("{:?}", eval(scan("`a`b!\"ab\"").unwrap()).unwrap()),
    format!("{:?}", KW::Noun(d1))
  );

  let k = K::SymbolArray(Series::new("a", ["a", "b"]).cast(&DataType::Categorical(None)).unwrap());
  let v = K::List(vec![K::Bool(1), K::Bool(0)]);
  let d1 = v_d_bang(k, v).unwrap();
  assert_eq!(
    format!("{:?}", eval(scan("`a`b!1 0").unwrap()).unwrap()),
    format!("{:?}", KW::Noun(d1))
  );

  let k = K::SymbolArray(Series::new("a", ["a"]).cast(&DataType::Categorical(None)).unwrap());
  let v = K::Bool(1);
  let d1 = v_d_bang(k, v).unwrap();
  assert_eq!(format!("{:?}", eval(scan("`a!1").unwrap()).unwrap()), format!("{:?}", KW::Noun(d1)));

  let k = K::SymbolArray(Series::new("a", ["a", "b", "c"]).cast(&DataType::Categorical(None)).unwrap());
  let v = K::List(vec![K::Bool(1), K::Bool(1), K::Bool(1)]);
  let d1 = v_d_bang(k, v).unwrap();
  assert_eq!(
    format!("{:?}", eval(scan("`a`b`c!1").unwrap()).unwrap()),
    format!("{:?}", KW::Noun(d1))
  );

}
