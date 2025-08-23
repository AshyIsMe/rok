use std::collections::HashMap;
use std::fs::{self, File};

use polars::prelude::*;
use roklang::*;

use roklang::KW::*;

fn k_eval(s: &str) -> K {
  let mut env = Env { names: HashMap::new(), parent: None };

  let r = eval(&mut env, scan(s).unwrap()).unwrap().unwrap_noun();
  println!("k_eval({}) = {}", s, r);
  r
}
fn k_evals(s: &str) -> String { format!("{}", k_eval(s)) }

#[test]
fn test_scan() {
  assert_eq!(scan("1").unwrap(), vec![Noun(K::Bool(1u8))]);
  assert_eq!(scan("2").unwrap(), vec![Noun(K::Int(Some(2)))]);
  assert_eq!(scan("3.14").unwrap(), vec![Noun(K::Float(3.14))]);
  assert_eq!(scan("1 0 1 0 1").unwrap(), vec![Noun(K::BoolArray(arr!([1, 0, 1, 0, 1u8])))]);
  assert_eq!(scan("1 2 3").unwrap(), vec![Noun(K::IntArray(arr!([1, 2, 3i64])))]);
  assert_eq!(scan("1 2 3.14").unwrap(), vec![Noun(K::FloatArray(arr!([1., 2., 3.14])))]);
  assert_eq!(
    scan("1-1").unwrap(),
    vec![Noun(K::Bool(1)), KW::Verb { name: "-".to_string() }, Noun(K::Bool(1))]
  );
  assert_eq!(
    scan("0N-1").unwrap(),
    vec![Noun(K::Int(None)), KW::Verb { name: "-".to_string() }, Noun(K::Bool(1))]
  );
  // This works but NAN != NAN so the assert fails
  // assert_eq!(
  //   scan("0n-1").unwrap(),
  //   vec![Noun(K::Float(f64::NAN)), KW::Verb { name: "-".to_string() }, Noun(K::Bool(1))]
  // );
}

#[test]
fn test_scan_lambda() {
  let tokens = vec![
    KW::Noun(K::Name("x".into())),
    KW::Verb { name: "+".to_string() },
    KW::Noun(K::Name("y".into())),
  ];
  let f = KW::Function { body: tokens, args: vec!["x".to_string(), "y".to_string()], adverb: None };
  assert_eq!(scan("{x+y}").unwrap(), vec![f]);
}

#[test]
fn test_scan_exprs() {
  let tokens = vec![vec![KW::Noun(K::Name("x".into()))], vec![KW::Noun(K::Name("y".into()))]];
  let e = KW::Exprs(tokens);
  assert_eq!(scan("[x;y]").unwrap(), vec![e]);
}

#[test]
fn test_scan_cond() {
  let tokens = vec![
    vec![KW::Noun(K::Bool(1))],
    vec![KW::Noun(K::Int(Some(2)))],
    vec![KW::Noun(K::Int(Some(3)))],
  ];
  let e = KW::Cond(tokens);
  assert_eq!(scan("$[1;2;3]").unwrap(), vec![e]);
}

#[test]
fn test_split_on() {
  let tokens = vec![
    // KW::CondStart, // don't pass open token through
    KW::Noun(K::Bool(1)),
    KW::SC,
    KW::Noun(K::Int(Some(2))),
    KW::SC,
    KW::Noun(K::Int(Some(3))),
    KW::RB,
    KW::Noun(K::Bool(1)),
  ];
  assert_eq!(
    split_on(tokens, KW::SC, KW::RB),
    Ok((
      vec![
        vec![KW::Noun(K::Bool(1))],
        vec![KW::Noun(K::Int(Some(2)))],
        vec![KW::Noun(K::Int(Some(3)))]
      ],
      vec![KW::Noun(K::Bool(1)),]
    ))
  );

  let tokens = vec![
    // KW::CondStart, // don't pass open token through
    KW::Noun(K::Bool(1)),
    KW::SC,
    KW::LB,
    KW::Noun(K::Int(Some(2))),
    KW::SC,
    KW::Noun(K::Int(Some(3))),
    KW::RB,
    KW::SC,
    KW::Noun(K::Int(Some(4))),
    KW::RB,
    KW::Noun(K::Bool(1)),
  ];
  assert_eq!(
    split_on(tokens, KW::SC, KW::RB),
    Ok((
      vec![
        vec![KW::Noun(K::Bool(1))],
        vec![KW::LB, KW::Noun(K::Int(Some(2))), KW::SC, KW::Noun(K::Int(Some(3))), KW::RB],
        vec![KW::Noun(K::Int(Some(4)))]
      ],
      vec![KW::Noun(K::Bool(1)),]
    ))
  );
}

#[test]
fn test_eval() {
  let mut env = Env { names: HashMap::new(), parent: None };

  assert_eq!(
    scan("2 + 2").unwrap(),
    vec![Noun(K::Int(Some(2))), Verb { name: "+".to_string() }, Noun(K::Int(Some(2)))]
  );
  assert_eq!(eval(&mut env, scan("2 + 2").unwrap()).unwrap(), Noun(K::Int(Some(4))));
  assert_eq!(eval(&mut env, scan("2 + 2 + 2").unwrap()).unwrap(), Noun(K::Int(Some(6))));

  assert_eq!(
    scan("1.0 + 2.0").unwrap(),
    vec![Noun(K::Float(1.0)), Verb { name: "+".to_string() }, Noun(K::Float(2.0))]
  );
  assert_eq!(eval(&mut env, scan("1.0 + 2.0").unwrap()).unwrap(), Noun(K::Float(3.0)));
  assert_eq!(eval(&mut env, scan("1.0 - 2.0").unwrap()).unwrap(), Noun(K::Float(-1.0)));
  assert_eq!(eval(&mut env, scan("2.0 * 2.0").unwrap()).unwrap(), Noun(K::Float(4.0)));
  assert_eq!(eval(&mut env, scan("10 % 2").unwrap()).unwrap(), Noun(K::Int(Some(5))));

  assert_eq!(
    eval(&mut env, scan("1 0 1 0 1 + 2").unwrap()).unwrap(),
    Noun(K::IntArray(arr!([3, 2, 3, 2, 3])))
  );

  // assert_eq!(eval(&mut env, scan("1 2 3 + 4 5 6").unwrap()).unwrap(), Noun(K::IntArray(arr!([5, 7, 9]))));
}

#[test]
fn test_parens() {
  let mut env = Env { names: HashMap::new(), parent: None };

  assert_eq!(scan("(1)").unwrap(), vec![KW::LP, Noun(K::Bool(1)), KW::RP]);
  assert_eq!(eval(&mut env, scan("(1)").unwrap()).unwrap(), Noun(K::Bool(1)));
  assert_eq!(eval(&mut env, scan("(1+1)").unwrap()).unwrap(), Noun(K::Int(Some(2))));
  assert_eq!(eval(&mut env, scan("(1+1)+3").unwrap()).unwrap(), Noun(K::Int(Some(5))));
  assert_eq!(eval(&mut env, scan("(1*1)+(3*3)").unwrap()).unwrap(), Noun(K::Int(Some(10))));
  assert_eq!(
    eval(&mut env, scan("((2 + 0 + 0 + 1 - 1) * (3*1)) + 1").unwrap()).unwrap(),
    Noun(K::Int(Some(7)))
  );
}

#[test]
fn test_array_maths() {
  let mut env = Env { names: HashMap::new(), parent: None };
  // TODO: Why do these fail the assert_eq even though they look correctly matched? (pointers vs values maybe?)
  // assert_eq!(eval(&mut env, scan("3.14 + 1 2 3").unwrap()).unwrap(), Noun(K::FloatArray(arr!([4.14, 5.14, 6.14]))));
  // assert_eq!(eval(&mut env, scan("3.14 + 1.0 2.0 3.0").unwrap()).unwrap(), Noun(K::FloatArray(arr!([4.14, 5.14, 6.14]))));
  assert_eq!(
    format!("{:?}", eval(&mut env, scan("3.14 + 1 2 3").unwrap()).unwrap()),
    format!("{:?}", Noun(K::FloatArray(arr!([4.14, 5.14, 6.14]))))
  );
  assert_eq!(
    format!("{:?}", eval(&mut env, scan("3.14 + 1.0 2.0 3.0").unwrap()).unwrap()),
    format!("{:?}", Noun(K::FloatArray(arr!([4.14, 5.14, 6.14]))))
  );

  assert_eq!(
    format!("{:?}", eval(&mut env, scan("1.1 2.1 + 1").unwrap()).unwrap()),
    format!("{:?}", Noun(K::FloatArray(arr!([2.1, 3.1]))))
  );
  assert_eq!(
    format!("{:?}", eval(&mut env, scan("1.1 2.1 + 2").unwrap()).unwrap()),
    format!("{:?}", Noun(K::FloatArray(arr!([3.1, 4.1]))))
  );
  assert_eq!(
    format!("{:?}", eval(&mut env, scan("1.1 2.1 + 2.0").unwrap()).unwrap()),
    format!("{:?}", Noun(K::FloatArray(arr!([3.1, 4.1]))))
  );
  assert_eq!(
    format!("{:?}", eval(&mut env, scan("1.1 2.1 + 1 1").unwrap()).unwrap()),
    format!("{:?}", Noun(K::FloatArray(arr!([2.1, 3.1]))))
  );
  assert_eq!(
    format!("{:?}", eval(&mut env, scan("1.1 2.1 + 2 2").unwrap()).unwrap()),
    format!("{:?}", Noun(K::FloatArray(arr!([3.1, 4.1]))))
  );
  assert_eq!(
    format!("{:?}", eval(&mut env, scan("1.1 2.1 + 2.0 2.0").unwrap()).unwrap()),
    format!("{:?}", Noun(K::FloatArray(arr!([3.1, 4.1]))))
  );

  assert_eq!(
    format!("{:?}", eval(&mut env, scan("1.1 2.1 - 1").unwrap()).unwrap()),
    format!("{:?}", Noun(K::FloatArray(arr!([0.1, 1.1]))))
  );
  assert_eq!(
    format!("{:?}", eval(&mut env, scan("1.1 2.1 - 2").unwrap()).unwrap()),
    format!("{:?}", Noun(K::FloatArray(arr!([-0.9, 0.1]))))
  );
  assert_eq!(
    format!("{:?}", eval(&mut env, scan("1.1 2.1 - 2.0").unwrap()).unwrap()),
    format!("{:?}", Noun(K::FloatArray(arr!([-0.9, 0.1]))))
  );
  assert_eq!(
    format!("{:?}", eval(&mut env, scan("1.1 2.1 - 1 1").unwrap()).unwrap()),
    format!("{:?}", Noun(K::FloatArray(arr!([0.1, 1.1]))))
  );
  assert_eq!(
    format!("{:?}", eval(&mut env, scan("1.1 2.1 - 2 2").unwrap()).unwrap()),
    format!("{:?}", Noun(K::FloatArray(arr!([-0.9, 0.1]))))
  );
  assert_eq!(
    format!("{:?}", eval(&mut env, scan("1.1 2.1 - 2.0 2.0").unwrap()).unwrap()),
    format!("{:?}", Noun(K::FloatArray(arr!([-0.9, 0.1]))))
  );

  assert_eq!(
    format!("{:?}", eval(&mut env, scan("1.1 2.1 * 1").unwrap()).unwrap()),
    format!("{:?}", Noun(K::FloatArray(arr!([1.1, 2.1]))))
  );
  assert_eq!(
    format!("{:?}", eval(&mut env, scan("1.1 2.1 * 2").unwrap()).unwrap()),
    format!("{:?}", Noun(K::FloatArray(arr!([2.2, 4.2]))))
  );
  assert_eq!(
    format!("{:?}", eval(&mut env, scan("1.1 2.1 * 2.0").unwrap()).unwrap()),
    format!("{:?}", Noun(K::FloatArray(arr!([2.2, 4.2]))))
  );
  assert_eq!(
    format!("{:?}", eval(&mut env, scan("1.1 2.1 * 1 1").unwrap()).unwrap()),
    format!("{:?}", Noun(K::FloatArray(arr!([1.1, 2.1]))))
  );
  assert_eq!(
    format!("{:?}", eval(&mut env, scan("1.1 2.1 * 2 2").unwrap()).unwrap()),
    format!("{:?}", Noun(K::FloatArray(arr!([2.2, 4.2]))))
  );
  assert_eq!(
    format!("{:?}", eval(&mut env, scan("1.1 2.1 * 2.0 2.0").unwrap()).unwrap()),
    format!("{:?}", Noun(K::FloatArray(arr!([2.2, 4.2]))))
  );

  assert_eq!(
    format!("{:?}", eval(&mut env, scan("1.1 2.1 % 1").unwrap()).unwrap()),
    format!("{:?}", Noun(K::FloatArray(arr!([1.1, 2.1]))))
  );
  assert_eq!(
    format!("{:?}", eval(&mut env, scan("1.1 2.1 % 2").unwrap()).unwrap()),
    format!("{:?}", Noun(K::FloatArray(arr!([0.55, 1.05]))))
  );
  assert_eq!(
    format!("{:?}", eval(&mut env, scan("1.1 2.1 % 2.0").unwrap()).unwrap()),
    format!("{:?}", Noun(K::FloatArray(arr!([0.55, 1.05]))))
  );
  assert_eq!(
    format!("{:?}", eval(&mut env, scan("1.1 2.1 % 1 1").unwrap()).unwrap()),
    format!("{:?}", Noun(K::FloatArray(arr!([1.1, 2.1]))))
  );
  assert_eq!(
    format!("{:?}", eval(&mut env, scan("1.1 2.1 % 2 2").unwrap()).unwrap()),
    format!("{:?}", Noun(K::FloatArray(arr!([0.55, 1.05]))))
  );
  assert_eq!(
    format!("{:?}", eval(&mut env, scan("1.1 2.1 % 2.0 2.0").unwrap()).unwrap()),
    format!("{:?}", Noun(K::FloatArray(arr!([0.55, 1.05]))))
  );
}

#[test]
fn test_iota() {
  let mut env = Env { names: HashMap::new(), parent: None };
  assert_eq!(
    format!("{:?}", eval(&mut env, scan("! 4").unwrap()).unwrap()),
    format!("{:?}", Noun(K::IntArray(arr!([0, 1, 2, 3i64]))))
  );
}

#[test]
fn test_strings() {
  let mut env = Env { names: HashMap::new(), parent: None };
  assert_eq!(
    format!("{:?}", eval(&mut env, scan("\"a\"").unwrap()).unwrap()),
    format!("{:?}", Noun(K::Char('a')))
  );
  assert_eq!(
    format!("{:?}", eval(&mut env, scan("\"\"").unwrap()).unwrap()),
    format!("{:?}", Noun(K::CharArray("".to_string())))
  );
  assert_eq!(
    format!("{:?}", eval(&mut env, scan("\"abcABC\"").unwrap()).unwrap()),
    format!("{:?}", Noun(K::CharArray("abcABC".to_string())))
  );
}

#[test]
fn test_symbols() {
  let mut env = Env { names: HashMap::new(), parent: None };
  assert_eq!(
    format!("{:?}", eval(&mut env, scan("`a").unwrap()).unwrap()),
    format!("{:?}", Noun(K::Symbol("a".to_string())))
  );
  assert_eq!(
    format!("{:?}", eval(&mut env, scan("`abc ").unwrap()).unwrap()),
    format!("{:?}", Noun(K::Symbol("abc".to_string())))
  );
  assert_eq!(
    format!("{:?}", eval(&mut env, scan("`a`b`c").unwrap()).unwrap()),
    format!(
      "{:?}",
      Noun(K::SymbolArray(
        Series::new("a", ["a", "b", "c"])
          .cast(&DataType::Categorical(None, CategoricalOrdering::Lexical))
          .unwrap()
      ))
    )
  );
  assert_eq!(
    format!("{:?}", eval(&mut env, scan("`a`b`c`").unwrap()).unwrap()),
    format!(
      "{:?}",
      Noun(K::SymbolArray(
        Series::new("a", ["a", "b", "c", ""])
          .cast(&DataType::Categorical(None, CategoricalOrdering::Lexical))
          .unwrap()
      ))
    )
  );
  assert_eq!(
    format!("{:?}", eval(&mut env, scan("`a `b `c").unwrap()).unwrap()),
    format!(
      "{:?}",
      Noun(K::SymbolArray(
        Series::new("a", ["a", "b", "c"])
          .cast(&DataType::Categorical(None, CategoricalOrdering::Lexical))
          .unwrap()
      ))
    )
  );
  assert_eq!(
    format!("{:?}", eval(&mut env, scan("`a ` `b `c").unwrap()).unwrap()),
    format!(
      "{:?}",
      Noun(K::SymbolArray(
        Series::new("a", ["a", "", "b", "c"])
          .cast(&DataType::Categorical(None, CategoricalOrdering::Lexical))
          .unwrap()
      ))
    )
  );
}

#[test]
fn test_quoted_symbols() {
  let mut env = Env { names: HashMap::new(), parent: None };
  let res = eval(&mut env, scan("`\"abc def\"").unwrap()).unwrap();
  println!("{:?}", res);
  assert_eq!(format!("{:?}", res), format!("{:?}", Noun(K::Symbol("abc def".to_string()))));

  let res = eval(&mut env, scan("`a`\"abc def\"").unwrap()).unwrap();
  println!("{:?}", res);
  assert_eq!(
    format!("{:?}", res),
    format!(
      "{:?}",
      Noun(K::SymbolArray(
        Series::new("a", ["a", "abc def"])
          .cast(&DataType::Categorical(None, CategoricalOrdering::Lexical))
          .unwrap()
      ))
    )
  );

  let res = eval(&mut env, scan("`a`\"abc def\"`\"foo!bar_baz\"").unwrap()).unwrap();
  println!("{:?}", res);
  assert_eq!(
    format!("{:?}", res),
    format!(
      "{:?}",
      Noun(K::SymbolArray(
        Series::new("a", ["a", "abc def", "foo!bar_baz"])
          .cast(&DataType::Categorical(None, CategoricalOrdering::Lexical))
          .unwrap()
      ))
    )
  );
}

#[test]
fn test_length_errors() {
  let mut env = Env { names: HashMap::new(), parent: None };
  assert_eq!(eval(&mut env, scan("1 2 3 + 4 5").unwrap()), Err::<KW, &'static str>("length"));
  assert_eq!(eval(&mut env, scan("1 2 + 3 4 5").unwrap()), Err::<KW, &'static str>("length"));
  assert_eq!(eval(&mut env, scan("1.0 2.0 + 3 4 5").unwrap()), Err::<KW, &'static str>("length"));
  assert_eq!(eval(&mut env, scan("1 2 3 - 4 5").unwrap()), Err::<KW, &'static str>("length"));
  assert_eq!(eval(&mut env, scan("1 2 - 3 4 5").unwrap()), Err::<KW, &'static str>("length"));
  assert_eq!(eval(&mut env, scan("1.0 2.0 - 3 4 5").unwrap()), Err::<KW, &'static str>("length"));
  assert_eq!(eval(&mut env, scan("1 2 3 * 4 5").unwrap()), Err::<KW, &'static str>("length"));
  assert_eq!(eval(&mut env, scan("1 2 * 3 4 5").unwrap()), Err::<KW, &'static str>("length"));
  assert_eq!(eval(&mut env, scan("1.0 2.0 * 3 4 5").unwrap()), Err::<KW, &'static str>("length"));
  assert_eq!(eval(&mut env, scan("1 2 3 % 4 5").unwrap()), Err::<KW, &'static str>("length"));
  assert_eq!(eval(&mut env, scan("1 2 % 3 4 5").unwrap()), Err::<KW, &'static str>("length"));
  assert_eq!(eval(&mut env, scan("1.0 2.0 % 3 4 5").unwrap()), Err::<KW, &'static str>("length"));
}

#[test]
fn test_lists() {
  let mut env = Env { names: HashMap::new(), parent: None };
  assert_eq!(
    format!("{:?}", eval(&mut env, scan("(1 2; 3 4)").unwrap()).unwrap()),
    format!(
      "{:?}",
      Noun(K::List(vec![K::IntArray(arr!([1, 2i64])), K::IntArray(arr!([3, 4i64]))]))
    )
  );
  assert_eq!(
    format!("{:?}", eval(&mut env, scan("(1;\"a\";2)").unwrap()).unwrap()),
    format!("{:?}", Noun(K::List(vec![K::Bool(1u8), K::Char('a'), K::Int(Some(2))])))
  );
  assert_eq!(
    format!("{:?}", eval(&mut env, scan("(1+1;2+2)").unwrap()).unwrap()),
    format!("{:?}", Noun(K::IntArray(arr!([2, 4i64]))))
  );

  assert_eq!(
    format!("{:?}", eval(&mut env, scan("(1;2;3)").unwrap()).unwrap()),
    format!("{:?}", Noun(K::IntArray(arr!([1, 2, 3i64]))))
  );
  assert_eq!(
    format!("{:?}", eval(&mut env, scan("(1;0;1)").unwrap()).unwrap()),
    format!("{:?}", Noun(K::BoolArray(arr!([1, 0, 1u8]))))
  );
  assert_eq!(
    format!("{:?}", eval(&mut env, scan("(1;0;1.5)").unwrap()).unwrap()),
    format!("{:?}", Noun(K::FloatArray(arr!([1.0, 0.0, 1.5f64]))))
  );
  assert_eq!(
    format!("{:?}", eval(&mut env, scan("(\"a\";\"b\";\"c\")").unwrap()).unwrap()),
    format!("{:?}", Noun(K::CharArray("abc".to_string())))
  );
  // TODO SymbolArray promotion
  // assert_eq!(
  //   format!("{:?}", eval(&mut env, scan("(`a;`b;`c)").unwrap()).unwrap()),
  //   format!(
  //     "{:?}",
  //     Noun(K::SymbolArray(
  //       Series::new("a", ["a", "b", "c"]).cast(&DataType::Categorical(None)).unwrap()
  //     ))
  //   )
  // );
}

#[test]
fn test_list_maths() {
  let mut env = Env { names: HashMap::new(), parent: None };
  assert_eq!(
    format!("{:?}", eval(&mut env, scan("1 + (1 2; 3 4)").unwrap()).unwrap()),
    format!(
      "{:?}",
      Noun(K::List(vec![K::IntArray(arr!([2, 3i64])), K::IntArray(arr!([4, 5i64]))]))
    )
  );
}

#[test]
fn test_dict() {
  let mut env = Env { names: HashMap::new(), parent: None };
  let k = K::SymbolArray(
    Series::new("a", ["a", "b", "c"])
      .cast(&DataType::Categorical(None, CategoricalOrdering::Lexical))
      .unwrap(),
  );
  let v = K::List(vec![K::Bool(1u8), K::Int(Some(42)), K::Float(3.14)]);
  let d1 = v_makedict(k, v);
  println!("{:?}", d1);

  let k = K::SymbolArray(
    Series::new("a", ["a", "b", "c"])
      .cast(&DataType::Categorical(None, CategoricalOrdering::Lexical))
      .unwrap(),
  );
  let v = K::List(vec![K::Bool(1u8)]);
  let d2 = v_makedict(k, v);
  println!("{:?}", d2);

  let k = K::SymbolArray(
    Series::new("a", ["a", "b"])
      .cast(&DataType::Categorical(None, CategoricalOrdering::Lexical))
      .unwrap(),
  );
  let v = K::List(vec![K::Char('a'), K::Int(Some(42))]);
  let d1 = v_makedict(k, v).unwrap();
  assert_eq!(
    format!("{:?}", eval(&mut env, scan("`a`b!(\"a\";42)").unwrap()).unwrap()),
    format!("{:?}", KW::Noun(d1))
  );

  let k = K::SymbolArray(
    Series::new("a", ["a", "b"])
      .cast(&DataType::Categorical(None, CategoricalOrdering::Lexical))
      .unwrap(),
  );
  let v = K::List(vec![K::Int(Some(1)), K::Int(Some(2))]);
  let d1 = v_makedict(k, v).unwrap();
  assert_eq!(
    format!("{:?}", eval(&mut env, scan("`a`b!1 2").unwrap()).unwrap()),
    format!("{:?}", KW::Noun(d1))
  );

  let k = K::SymbolArray(
    Series::new("a", ["a", "b"])
      .cast(&DataType::Categorical(None, CategoricalOrdering::Lexical))
      .unwrap(),
  );
  let v = K::List(vec![K::Int(Some(1)), K::Int(None)]);
  let d1 = v_makedict(k, v).unwrap();
  assert_eq!(
    format!("{:?}", eval(&mut env, scan("`a`b!1 0N").unwrap()).unwrap()),
    format!("{:?}", KW::Noun(d1))
  );

  let k = K::SymbolArray(
    Series::new("a", ["a", "b"])
      .cast(&DataType::Categorical(None, CategoricalOrdering::Lexical))
      .unwrap(),
  );
  let v = K::List(vec![K::Float(1.5), K::Float(2.5)]);
  let d1 = v_makedict(k, v).unwrap();
  assert_eq!(
    format!("{:?}", eval(&mut env, scan("`a`b!1.5 2.5").unwrap()).unwrap()),
    format!("{:?}", KW::Noun(d1))
  );

  let k = K::SymbolArray(
    Series::new("a", ["a", "b"])
      .cast(&DataType::Categorical(None, CategoricalOrdering::Lexical))
      .unwrap(),
  );
  let v = K::List(vec![K::Char('a'), K::Char('b')]);
  let d1 = v_makedict(k, v).unwrap();
  assert_eq!(
    format!("{:?}", eval(&mut env, scan("`a`b!\"ab\"").unwrap()).unwrap()),
    format!("{:?}", KW::Noun(d1))
  );

  let k = K::SymbolArray(
    Series::new("a", ["a", "b"])
      .cast(&DataType::Categorical(None, CategoricalOrdering::Lexical))
      .unwrap(),
  );
  let v = K::List(vec![K::Bool(1), K::Bool(0)]);
  let d1 = v_makedict(k, v).unwrap();
  assert_eq!(
    format!("{:?}", eval(&mut env, scan("`a`b!1 0").unwrap()).unwrap()),
    format!("{:?}", KW::Noun(d1))
  );

  let k = K::SymbolArray(
    Series::new("a", ["a"])
      .cast(&DataType::Categorical(None, CategoricalOrdering::Lexical))
      .unwrap(),
  );
  let v = K::Bool(1);
  let d1 = v_makedict(k, v).unwrap();
  assert_eq!(
    format!("{:?}", eval(&mut env, scan("`a!1").unwrap()).unwrap()),
    format!("{:?}", KW::Noun(d1))
  );

  let k = K::SymbolArray(
    Series::new("a", ["a", "b", "c"])
      .cast(&DataType::Categorical(None, CategoricalOrdering::Lexical))
      .unwrap(),
  );
  let v = K::List(vec![K::Bool(1), K::Bool(1), K::Bool(1)]);
  let d1 = v_makedict(k, v).unwrap();
  assert_eq!(
    format!("{:?}", eval(&mut env, scan("`a`b`c!1").unwrap()).unwrap()),
    format!("{:?}", KW::Noun(d1))
  );
}

#[test]
fn test_dict_maths() {
  let mut env = Env { names: HashMap::new(), parent: None };

  // let k = K::SymbolArray(Series::new("a", ["a"]).cast(&DataType::Categorical(None)).unwrap());
  let k = K::Symbol("a".into());
  let v = K::IntArray(arr!([2, 3i64]));
  let d1 = v_makedict(k, v).unwrap();
  let d2 = eval(&mut env, scan("1 2 + `a!1").unwrap()).unwrap();
  assert_eq!(format!("{:?}", Noun(d1)), format!("{:?}", d2));

  let k = K::Symbol("a".into());
  let v = K::FloatArray(arr!([2.0, 3.0f64]));
  let d1 = v_makedict(k, v).unwrap();
  let d2 = eval(&mut env, scan("1.0 2.0 + `a!1").unwrap()).unwrap();
  assert_eq!(format!("{:?}", Noun(d1)), format!("{:?}", d2));

  let k = K::Symbol("a".into());
  let v = K::IntArray(arr!([2, 1, 2i64]));
  let d1 = v_makedict(k, v).unwrap();
  let d2 = eval(&mut env, scan("1 0 1 + `a!1").unwrap()).unwrap();
  assert_eq!(format!("{:?}", Noun(d1)), format!("{:?}", d2));

  let k = K::SymbolArray(
    Series::new("a", ["a", "b"])
      .cast(&DataType::Categorical(None, CategoricalOrdering::Lexical))
      .unwrap(),
  );
  let v = K::List(vec![K::IntArray(arr!([1, 2i64])), K::IntArray(arr!([2, 3i64]))]);
  let d1 = v_d_bang(k, v).unwrap();
  let d2 = eval(&mut env, scan("1 2 + `a`b!(0;1)").unwrap()).unwrap();
  assert_eq!(format!("{:?}", KW::Noun(d1)), format!("{:?}", d2));

  let k = K::SymbolArray(
    Series::new("a", ["a", "b", "c"])
      .cast(&DataType::Categorical(None, CategoricalOrdering::Lexical))
      .unwrap(),
  );
  let v = K::IntArray(arr!([2, 4, 3i64]));
  let d1 = v_d_bang(k, v).unwrap();
  let d2 = eval(&mut env, scan("(`a`b!1 2) + `a`b`c!1 2 3").unwrap()).unwrap();
  assert_eq!(format!("{:?}", KW::Noun(d1)), format!("{:?}", d2));

  let k = K::SymbolArray(
    Series::new("a", ["a", "b", "c", "d"])
      .cast(&DataType::Categorical(None, CategoricalOrdering::Lexical))
      .unwrap(),
  );
  let v = K::IntArray(arr!([2, 4, 3, 4i64]));
  let d1 = v_d_bang(k, v).unwrap();
  let d2 = eval(&mut env, scan("(`a`b`c!1 2 3) + `a`b`d!1 2 4").unwrap()).unwrap();
  assert_eq!(format!("{:?}", KW::Noun(d1)), format!("{:?}", d2));
}

#[test]
fn test_table() {
  let mut env = Env { names: HashMap::new(), parent: None };
  let t1 = K::Table(DataFrame::new(vec![Series::new("a", [1, 2, 3i64])]).unwrap());
  let t2 = eval(&mut env, scan("+ `a!1 2 3").unwrap()).unwrap();
  println!("{:?}", t1);
  println!("{:?}", t2);
  assert_eq!(format!("{:?}", t2), format!("{:?}", KW::Noun(t1)));

  let t1 = K::Table(
    DataFrame::new(vec![Series::new("a", [1, 2, 3i64]), Series::new("b", [4, 5, 6i64])]).unwrap(),
  );
  let t2 = eval(&mut env, scan("+ `a`b!(1 2 3;4 5 6)").unwrap()).unwrap();
  println!("{:?}", t1);
  println!("{:?}", t2);
  assert_eq!(format!("{:?}", t2), format!("{:?}", KW::Noun(t1)));

  let s1 = K::IntArray(Series::new("a", [1, 2, 3i64]));
  let s2 = eval(&mut env, scan("(+ `a`b!(1 2 3;4 5 6))@`a").unwrap()).unwrap();
  println!("{:?}", s1);
  println!("{:?}", s2);
  assert_eq!(format!("{:?}", s2), format!("{:?}", KW::Noun(s1)));

  let s1 = eval(&mut env, scan("(1 2 3;4 5 6)").unwrap()).unwrap();
  let s2 = eval(&mut env, scan("(+ `a`b!(1 2 3;4 5 6))@`a`b").unwrap()).unwrap();
  println!("{:?}", s1);
  println!("{:?}", s2);
  assert_eq!(s1, s2);
}

#[test]
fn test_table_flip() {
  let mut env = Env { names: HashMap::new(), parent: None };

  let _d1 = eval(&mut env, scan("d:`a`b!(1 2 3;1 0 1)").unwrap()).unwrap();
  let _t1 = eval(&mut env, scan("t:+d").unwrap()).unwrap();
  let d2 = eval(&mut env, scan("+t").unwrap()).unwrap();
  let k = K::SymbolArray(
    Series::new("a", ["a", "b"])
      .cast(&DataType::Categorical(None, CategoricalOrdering::Lexical))
      .unwrap(),
  );
  let v = K::List(vec![
    K::IntArray(Series::new("a", [1, 2, 3i64])),
    K::BoolArray(Series::new("b", [1, 0, 1u8])),
  ]);
  let d1 = KW::Noun(v_makedict(k, v).unwrap());
  assert_eq!(d1, d2)
}

#[ignore]
#[test]
fn test_table_flip2() {
  let mut env = Env { names: HashMap::new(), parent: None };

  let t1 = K::Table(
    DataFrame::new(vec![Series::new("a", [1, 2, 3i64]), Series::new("b", ["foo", "abcd", "yo"])])
      .unwrap(),
  );
  println!("t1: {:?}", t1);

  // let t2 = eval(&mut env, scan("+`a`b!(1 2 3; (\"foo\";\"abcd\";\"yo\"))").unwrap()).unwrap();
  let k = eval(&mut env, scan("k:`a`b").unwrap()).unwrap();
  println!("k: {:?}", k);
  let v = eval(&mut env, scan("v:(1 2 3; (\"foo\";\"abcd\";\"yo\"))").unwrap()).unwrap();
  println!("v: {:?}", v);
  let t2 = eval(&mut env, scan("t:+k!v").unwrap()).unwrap();
  println!("t2: {:?}", t2);
  // assert_eq!(format!("{:?}", KW::Noun(t1)), format!("{:?}", t2));
  assert_eq!(KW::Noun(t1), t2);
  // assert!(true)

  // todo!("++`a`b!(1 2 3; (\"foo\";\"abcd\";\"yo\"))")
}

#[test]
fn test_table_reader() {
  let mut env = Env { names: HashMap::new(), parent: None };
  let mut df =
    DataFrame::new(vec![Series::new("a", [1, 2, 3i64]), Series::new("b", [4, 5, 6i64])]).unwrap();
  let mut file = File::create("test.csv").expect("could not create file");
  let _ = CsvWriter::new(&mut file).include_header(true).with_separator(b',').finish(&mut df);

  let t1 = K::Table(df.clone());
  let t2 = eval(&mut env, scan("2:`test.csv").unwrap()).unwrap();
  println!("{:?}", t1);
  println!("{:?}", t2);
  assert_eq!(format!("{:?}", t2), format!("{:?}", KW::Noun(t1.clone())));
  fs::remove_file("test.csv").unwrap();

  let file = File::create("test.parquet").expect("could not create file");
  let _ = ParquetWriter::new(file).finish(&mut df);
  let t2 = eval(&mut env, scan("2:`test.parquet").unwrap()).unwrap();
  println!("{:?}", t1);
  println!("{:?}", t2);
  assert_eq!(format!("{:?}", t2), format!("{:?}", KW::Noun(t1)));
  fs::remove_file("test.parquet").unwrap();
}

#[test]
fn test_table_maths() {
  let mut env = Env { names: HashMap::new(), parent: None };
  let t1 = eval(&mut env, scan("2 * + `a`b!(1 2 3;4 5 6)").unwrap()).unwrap();
  let t2 = eval(&mut env, scan("+ `a`b!(2 4 6;8 10 12)").unwrap()).unwrap();
  println!("{:?}", t1);
  println!("{:?}", t2);
  assert_eq!(format!("{:?}", t1), format!("{:?}", t2));

  let t1 = eval(&mut env, scan("3.14 * + `a`b!(1 2 3;4 5 6)").unwrap()).unwrap();
  let t2 = eval(&mut env, scan("+ `a`b!(3.14 6.28 9.42;12.56 15.7 18.84)").unwrap()).unwrap();
  println!("{:?}", t1);
  println!("{:?}", t2);
  assert_eq!(format!("{:?}", t1), format!("{:?}", t2));

  let t1 = eval(&mut env, scan("2 + + `a`b!(1 2 3;4 5 6)").unwrap()).unwrap();
  let t2 = eval(&mut env, scan("+ `a`b!(3 4 5;6 7 8)").unwrap()).unwrap();
  println!("{:?}", t1);
  println!("{:?}", t2);
  assert_eq!(format!("{:?}", t1), format!("{:?}", t2));

  let t1 = eval(&mut env, scan("(+ `a`b!(1 2 3;4 5 6)) * 2").unwrap()).unwrap();
  let t2 = eval(&mut env, scan("+ `a`b!(2 4 6;8 10 12)").unwrap()).unwrap();
  assert_eq!(format!("{:?}", t1), format!("{:?}", t2));
}

#[test]
fn test_names() {
  let mut env = Env { names: HashMap::new(), parent: None };

  let n1 = scan("abc def foo.bar").unwrap();
  let n2 = vec![
    KW::Noun(K::Name("abc".into())),
    KW::Noun(K::Name("def".into())),
    KW::Noun(K::Name("foo.bar".into())),
  ];
  assert_eq!(format!("{:?}", n1), format!("{:?}", n2));

  assert_eq!(eval(&mut env, scan("a:42").unwrap()).unwrap(), Noun(K::Int(Some(42))));
  assert_eq!(eval(&mut env, scan("a+a").unwrap()).unwrap(), Noun(K::Int(Some(84))));
  assert_eq!(eval(&mut env, scan("a:a+a:2").unwrap()).unwrap(), Noun(K::Int(Some(4))));
}

#[test]
fn test_named_verbs() {
  let mut env = Env { names: HashMap::new(), parent: None };

  assert_eq!(eval(&mut env, scan("p:+").unwrap()).unwrap(), Verb { name: "+".into() });
  assert_eq!(eval(&mut env, scan("2 p 2").unwrap()).unwrap(), Noun(K::Int(Some(4))));
}

#[test]
fn test_fold() {
  let mut env = Env { names: HashMap::new(), parent: None };

  assert_eq!(eval(&mut env, scan("+/ 1 2 3").unwrap()).unwrap(), Noun(K::Int(Some(6))));
  assert_eq!(eval(&mut env, scan("2 +/ 1 2 3").unwrap()).unwrap(), Noun(K::Int(Some(8))));

  assert_eq!(eval(&mut env, scan("+/ 1 2 3 4").unwrap()).unwrap(), Noun(K::Int(Some(10))));
  assert_eq!(eval(&mut env, scan("*/ 1 2 3 4").unwrap()).unwrap(), Noun(K::Int(Some(24))));

  assert_eq!(eval(&mut env, scan("2 +/ 1 2 3 4").unwrap()).unwrap(), Noun(K::Int(Some(12))));
  assert_eq!(eval(&mut env, scan("2 */ 1 2 3 4").unwrap()).unwrap(), Noun(K::Int(Some(48))));

  assert_eq!(eval(&mut env, scan("{x + y}/ 1 2 3 4").unwrap()).unwrap(), Noun(K::Int(Some(10))));

  assert_eq!(
    eval(&mut env, scan("+/(1 2 3; 4 5 6)").unwrap()).unwrap(),
    Noun(K::IntArray(Series::new("a", [5, 7, 9i64])))
  );
  assert_eq!(
    eval(&mut env, scan("{x+y}/(1 2 3; 4 5 6)").unwrap()).unwrap(),
    Noun(K::IntArray(Series::new("a", [5, 7, 9i64])))
  );
}

#[test]
fn test_v_scan() {
  let mut env = Env { names: HashMap::new(), parent: None };

  assert_eq!(
    eval(&mut env, scan("+\\ 1 2 3").unwrap()).unwrap(),
    Noun(K::IntArray(Series::new("a", [1, 3, 6i64])))
  );
}

#[test]
fn test_fixedpoint() {
  let mut env = Env { names: HashMap::new(), parent: None };

  assert_eq!(eval(&mut env, scan("{0|x-1}/5").unwrap()).unwrap(), Noun(K::Int(Some(0))));

  assert_eq!(eval(&mut env, scan("+//(1 2 3; 4 5 6)").unwrap()).unwrap(), Noun(K::Int(Some(21))));

  // "+//((1 2 3;4 5 6);42)" => 273
  assert_eq!(
    eval(&mut env, scan("+//((1 2 3;4 5 6);42)").unwrap()).unwrap(),
    Noun(K::Int(Some(273)))
  );
}

#[test]
fn test_scan_fixedpoint() {
  let mut env = Env { names: HashMap::new(), parent: None };

  assert_eq!(
    eval(&mut env, scan("{0|x-1}\\5").unwrap()).unwrap(),
    Noun(K::IntArray(Series::new("a", [5, 4, 3, 2, 1, 0i64])))
  );

  assert_eq!(
    eval(&mut env, scan("+/\\(1 2 3; 4 5 6)").unwrap()).unwrap(),
    // ((1 2 3;4 5 6);5 7 9;21)
    Noun(K::List(vec![
      K::List(vec![
        K::IntArray(Series::new("", [1, 2, 3i64])),
        K::IntArray(Series::new("", [4, 5, 6i64]))
      ]),
      K::IntArray(Series::new("", [5, 7, 9i64])),
      K::Int(Some(21))
    ]))
  );
}

#[test]
fn test_v_d_scan() {
  let mut env = Env { names: HashMap::new(), parent: None };

  assert_eq!(
    eval(&mut env, scan("3 +\\ 1 2 3").unwrap()).unwrap(),
    eval(&mut env, scan("4 6 9").unwrap()).unwrap(),
  );
}

#[test]
fn test_parse_functions() {
  // See https://estradajke.github.io/k9-simples/k9/User-Functions.html
  //
  let f = KW::Function {
    body: vec![
      KW::Noun(K::Int(Some(2))),
      KW::Verb { name: "*".to_string() },
      KW::Noun(K::Name("x".to_string())),
    ],
    args: vec!["x".to_string()],
    adverb: None,
  };
  let mut env = Env { names: HashMap::new(), parent: None };
  assert_eq!(eval(&mut env, scan("{2 * x}").unwrap()).unwrap(), f);

  let f = KW::Function {
    body: vec![KW::Exprs(vec![
      vec![
        KW::Noun(K::Name("a".to_string())),
        KW::Verb { name: ":".to_string() },
        KW::Noun(K::Int(Some(2))),
      ],
      vec![
        KW::Noun(K::Name("x".to_string())),
        KW::Verb { name: "*".to_string() },
        KW::Noun(K::Name("a".to_string())),
      ],
    ])],
    args: vec!["x".to_string()],
    adverb: None,
  };
  let mut env = Env { names: HashMap::new(), parent: None };
  assert_eq!(eval(&mut env, scan("{a:2;x*a}").unwrap()).unwrap(), f);

  let f = KW::Function {
    body: vec![KW::Exprs(vec![
      vec![
        KW::Noun(K::Name("a".to_string())),
        KW::Verb { name: ":".to_string() },
        KW::Noun(K::Int(Some(2))),
      ],
      vec![KW::Exprs(vec![vec![
        KW::Noun(K::Name("x".to_string())),
        KW::Verb { name: "*".to_string() },
        KW::Noun(K::Name("a".to_string())),
      ]])],
    ])],
    args: vec!["x".to_string()],
    adverb: None,
  };
  let mut env = Env { names: HashMap::new(), parent: None };
  assert_eq!(eval(&mut env, scan("{a:2;[x*a]}").unwrap()).unwrap(), f);
}

#[test]
fn test_functions() {
  // See https://estradajke.github.io/k9-simples/k9/User-Functions.html
  //
  let mut env = Env { names: HashMap::new(), parent: None };
  assert_eq!(eval(&mut env, scan("{2 * x} 2").unwrap()).unwrap(), Noun(K::Int(Some(4))));

  eval(&mut env, scan("f:{2 * x}").unwrap()).unwrap();
  assert_eq!(eval(&mut env, scan("f 2").unwrap()).unwrap(), Noun(K::Int(Some(4))));

  assert_eq!(
    eval(&mut env, scan("{x*2}!5").unwrap()).unwrap().unwrap_noun(),
    K::IntArray(arr!([0, 2, 4, 6, 8i64]))
  );
}

#[test]
fn test_function_local_vars() {
  // See https://estradajke.github.io/k9-simples/k9/User-Functions.html
  //
  let mut env = Env { names: HashMap::new(), parent: None };
  assert_eq!(eval(&mut env, scan("{a:2; a * x} 2").unwrap()).unwrap(), Noun(K::Int(Some(4))));
}

#[test]
fn test_expr_funcargs() {
  let mut env = Env { names: HashMap::new(), parent: None };

  eval(&mut env, scan("f:{[x;y]x+y}").unwrap()).unwrap();
  assert_eq!(eval(&mut env, scan("f[2;2]").unwrap()).unwrap(), Noun(K::Int(Some(4))));

  eval(&mut env, scan("f:{[a;b;c]a+b+c}").unwrap()).unwrap();
  assert_eq!(eval(&mut env, scan("f[2;2;2]").unwrap()).unwrap(), Noun(K::Int(Some(6))));

  // let mut env = Env { names: HashMap::new(), parent: None };
  // AA TODO a space between verb and arg list changes from verb[args] to verb (eval args) => verb(curry)list
  // assert_eq!(eval(&mut env, scan("+ [2;2]").unwrap()).unwrap(), CurryVerb("+", Noun(K::Int(Some(2)))));
}

#[test]
fn test_expr_order() {
  let mut env = Env { names: HashMap::new(), parent: None };

  // exprs in k are evaluated right to left but semicolon effectively acts as NewLine
  assert_eq!(eval(&mut env, scan("[a:2;a+2]").unwrap()).unwrap(), Noun(K::Int(Some(4))));
  // unbracketed exprs also should work
  assert_eq!(eval(&mut env, scan("b:2;b+2").unwrap()).unwrap(), Noun(K::Int(Some(4))));

  assert_eq!(eval(&mut env, scan("2 + [c:2;c]").unwrap()).unwrap(), Noun(K::Int(Some(4))));
}

#[ignore]
#[test]
fn test_expr() {
  // TODO these should work but seem low priority
  let mut env = Env { names: HashMap::new(), parent: None };
  assert_eq!(eval(&mut env, scan("[a:2;a] + 2").unwrap()).unwrap(), Noun(K::Int(Some(4))));
  assert_eq!(eval(&mut env, scan("2 + [a:2;a] + 0").unwrap()).unwrap(), Noun(K::Int(Some(4))));
}

#[test]
fn test_named_primitives() {
  let mut env = Env { names: HashMap::new(), parent: None };
  assert_eq!(eval(&mut env, scan("+[2;2]").unwrap()).unwrap(), Noun(K::Int(Some(4))));

  eval(&mut env, scan("p:+").unwrap()).unwrap();
  assert_eq!(eval(&mut env, scan("p[2;2]").unwrap()).unwrap(), Noun(K::Int(Some(4))));
}

#[test]
fn test_cond() {
  let mut env = Env { names: HashMap::new(), parent: None };
  let res = eval(&mut env, scan("$[1;`a;0;`b]").unwrap()).unwrap();
  println!("res: {:?}", res);
  assert_eq!(res, Noun(K::Symbol("a".to_string())));

  let res = eval(&mut env, scan("$[0;`a;0;`b;`c]").unwrap()).unwrap();
  println!("res: {:?}", res);
  assert_eq!(res, Noun(K::Symbol("c".to_string())));
}

#[test]
fn test_equal() {
  let mut env = Env { names: HashMap::new(), parent: None };
  let res = eval(&mut env, scan("1 = 1").unwrap()).unwrap();
  println!("res: {:?}", res);
  assert_eq!(res, Noun(K::Bool(1)));

  let res = eval(&mut env, scan("1 2 3 = 1 2 3").unwrap()).unwrap();
  println!("res: {:?}", res);
  assert_eq!(res, Noun(K::BoolArray(arr!([1, 1, 1u8]))));

  let res = eval(&mut env, scan("1 = 1 2 3").unwrap()).unwrap();
  println!("res: {:?}", res);
  assert_eq!(res, Noun(K::BoolArray(arr!([1, 0, 0u8]))));

  let res = eval(&mut env, scan("(1;2;\"3\") = (1;2;\"3\")").unwrap()).unwrap();
  println!("res: {:?}", res);
  assert_eq!(res, Noun(K::BoolArray(arr!([1, 1, 1u8]))));

  let res = eval(&mut env, scan("(1;2;\"3\") = (1;2;\"a\")").unwrap()).unwrap();
  println!("res: {:?}", res);
  assert_eq!(res, Noun(K::BoolArray(arr!([1, 1, 0u8]))));

  let res = eval(&mut env, scan("(1;2;\"3\") = 1 2 3").unwrap()).unwrap();
  println!("res: {:?}", res);
  assert_eq!(res, Noun(K::BoolArray(arr!([1, 1, 0u8]))));

  let res = eval(&mut env, scan("(1;2;\"a\") = `a`b`c!(1;2;3)").unwrap()).unwrap();
  println!("res: {:?}", res);
  let k = K::SymbolArray(
    Series::new("a", ["a", "b", "c"])
      .cast(&DataType::Categorical(None, CategoricalOrdering::Lexical))
      .unwrap(),
  );
  let v = K::List(vec![K::Bool(1u8), K::Bool(1u8), K::Bool(0u8)]);
  let d1 = v_makedict(k, v).unwrap();
  assert_eq!(res, Noun(d1));

  let res = eval(&mut env, scan("(`a`b`c!(1;2;3)) = (1;2;\"a\")").unwrap()).unwrap();
  println!("res: {:?}", res);
  let k = K::SymbolArray(
    Series::new("a", ["a", "b", "c"])
      .cast(&DataType::Categorical(None, CategoricalOrdering::Lexical))
      .unwrap(),
  );
  let v = K::List(vec![K::Bool(1u8), K::Bool(1u8), K::Bool(0u8)]);
  let d1 = v_makedict(k, v).unwrap();
  assert_eq!(res, Noun(d1));

  let res = eval(&mut env, scan("1 2 3 = `a`b`c!(1;2;3)").unwrap()).unwrap();
  println!("res: {:?}", res);
  let k = K::SymbolArray(
    Series::new("a", ["a", "b", "c"])
      .cast(&DataType::Categorical(None, CategoricalOrdering::Lexical))
      .unwrap(),
  );
  let v = K::List(vec![K::Bool(1u8), K::Bool(1u8), K::Bool(1u8)]);
  let d1 = v_makedict(k, v).unwrap();
  assert_eq!(res, Noun(d1));

  let res = eval(&mut env, scan("`a`b`c!(1;2;3) = 1 2 3").unwrap()).unwrap();
  println!("res: {:?}", res);
  let k = K::SymbolArray(
    Series::new("a", ["a", "b", "c"])
      .cast(&DataType::Categorical(None, CategoricalOrdering::Lexical))
      .unwrap(),
  );
  let v = K::List(vec![K::Bool(1u8), K::Bool(1u8), K::Bool(1u8)]);
  let d1 = v_makedict(k, v).unwrap();
  assert_eq!(res, Noun(d1));

  let res = eval(&mut env, scan("`a`b`c!(1;2;3) = 1.0 2.0 3.0").unwrap()).unwrap();
  println!("res: {:?}", res);
  let k = K::SymbolArray(
    Series::new("a", ["a", "b", "c"])
      .cast(&DataType::Categorical(None, CategoricalOrdering::Lexical))
      .unwrap(),
  );
  let v = K::List(vec![K::Bool(1u8), K::Bool(1u8), K::Bool(1u8)]);
  let d1 = v_makedict(k, v).unwrap();
  assert_eq!(res, Noun(d1));
}

#[test]
fn test_unique() {
  let mut env = Env { names: HashMap::new(), parent: None };
  let res = eval(&mut env, scan("? 1 1 2 2 3").unwrap()).unwrap();
  println!("res: {:?}", res);
  assert_eq!(res, Noun(K::IntArray(arr!([1, 2, 3i64]))));

  let res1 = eval(&mut env, scan("? 1 1 0 0").unwrap()).unwrap();
  println!("res1: {:?}", res1);
  let res2 = eval(&mut env, scan("0 1").unwrap()).unwrap();
  assert_eq!(res1, res2);

  let res = eval(&mut env, scan("? 1 1 2 2 3.0").unwrap()).unwrap();
  println!("res: {:?}", res);
  assert_eq!(res, Noun(K::FloatArray(arr!([1.0, 2.0, 3.0f64]))));

  assert_eq!(
    eval(&mut env, scan("? `a`a`b`c`c").unwrap()).unwrap(),
    Noun(K::SymbolArray(
      Series::new("a", ["a", "b", "c"])
        .cast(&DataType::Categorical(None, CategoricalOrdering::Lexical))
        .unwrap()
    ))
  );

  // This works but the result is not sorted.
  // assert_eq!(
  //   eval(&mut env, scan("? \"aabbcc\"").unwrap()).unwrap(),
  //   Noun(K::CharArray(Series::new("", "abc").cast(&DataType::Utf8).unwrap()))
  // )
  let res = eval(&mut env, scan("? \"aabb\"").unwrap()).unwrap();
  let ab = Noun(K::CharArray("ab".to_string()));
  let ba = Noun(K::CharArray("ba".to_string()));
  assert!(res == ab || res == ba);
}

#[test]
fn test_array_indexing() {
  let mut env = Env { names: HashMap::new(), parent: None };
  let res = eval(&mut env, scan("1 2 3 4 @ 0").unwrap()).unwrap();
  assert_eq!(res, Noun(K::Int(Some(1i64))));
  let res = eval(&mut env, scan("1 2 3.14 4 @ 0").unwrap()).unwrap();
  assert_eq!(res, Noun(K::Float(1.0f64)));

  let res = eval(&mut env, scan("\"abc\" @ 0").unwrap()).unwrap();
  assert_eq!(res, Noun(K::Char('a')));

  let res = eval(&mut env, scan("\"abc\" @ 1").unwrap()).unwrap();
  assert_eq!(res, Noun(K::Char('b')));

  let res = eval(&mut env, scan("\"abc\" @ 2").unwrap()).unwrap();
  assert_eq!(res, Noun(K::Char('c')));

  let res = eval(&mut env, scan("\"abc\" @ 2 1 0").unwrap()).unwrap();
  assert_eq!(res, Noun(K::CharArray("cba".to_string())));

  let res = eval(&mut env, scan("1 2 3 4 @ 0 1 2").unwrap()).unwrap();
  assert_eq!(res, Noun(K::IntArray(arr!([1, 2, 3i64]))));

  let res = eval(&mut env, scan("1 2 3.14 4 @ 0 1 2").unwrap()).unwrap();
  assert_eq!(res, Noun(K::FloatArray(arr!([1.0, 2.0, 3.14f64]))));

  let res = eval(&mut env, scan("(`a`b!(1 2 3;\"abc\")) @ `a").unwrap()).unwrap();
  assert_eq!(res, Noun(K::IntArray(arr!([1, 2, 3i64]))));

  let res = eval(&mut env, scan("(`a`b!(1 2 3;\"abc\")) @ `b").unwrap()).unwrap();
  assert_eq!(res, Noun(K::CharArray("abc".to_string())));

  let res = eval(&mut env, scan("(`a`b!(1 2 3;1 0 1)) @ `b").unwrap()).unwrap();
  assert_eq!(res, Noun(K::BoolArray(arr!([1, 0, 1u8]))));

  let res = eval(&mut env, scan("(`a`b!(1 2 3;1 0 1)) @ `a`b").unwrap()).unwrap();
  assert_eq!(
    res,
    Noun(K::List(vec![K::IntArray(arr!([1, 2, 3i64])), K::BoolArray(arr!([1, 0, 1u8]))]))
  );

  let res = eval(&mut env, scan("(1 2 3; 3.14; `a) @ 0").unwrap()).unwrap();
  assert_eq!(res, Noun(K::IntArray(arr!([1, 2, 3i64]))));

  // (1;2;3 4)@0 1
  // (1;2;3 4)@2
  let res = eval(&mut env, scan("(1;2;3 4)@ 0").unwrap()).unwrap();
  assert_eq!(res, Noun(K::Bool(1)));

  let res = eval(&mut env, scan("(1;2;3 4)@ 0 1").unwrap()).unwrap();
  assert_eq!(res, Noun(K::IntArray(arr!([1, 2i64]))));

  let res = eval(&mut env, scan("(1;2;3 4)@ 2").unwrap()).unwrap();
  assert_eq!(res, Noun(K::IntArray(arr!([3, 4i64]))));

  assert_eq!(
    eval(&mut env, scan("(1;2;3 4)@ 42 42 42").unwrap()).unwrap(),
    eval(&mut env, scan("0N 0N 0N").unwrap()).unwrap()
  );
}

#[test]
fn test_vmod() {
  let mut env = Env { names: HashMap::new(), parent: None };

  assert_eq!(
    eval(&mut env, scan("2!1 2 3 4 5").unwrap()).unwrap(),
    Noun(K::IntArray(arr!([1, 0, 1, 0, 1i64])))
  );
  assert_eq!(
    eval(&mut env, scan("3!1 2 3 4 5").unwrap()).unwrap(),
    Noun(K::IntArray(arr!([1, 2, 0, 1, 2i64])))
  );
  assert_eq!(
    eval(&mut env, scan("2!1 2 3 4 5.5").unwrap()).unwrap(),
    Noun(K::FloatArray(arr!([1.0, 0.0, 1.0, 0.0, 1.5f64])))
  );
}

#[test]
fn test_reshape() {
  let mut env = Env { names: HashMap::new(), parent: None };

  assert_eq!(
    eval(&mut env, scan("3 3#1").unwrap()).unwrap(),
    Noun(K::List(vec![
      K::BoolArray(arr!([1, 1, 1u8])),
      K::BoolArray(arr!([1, 1, 1u8])),
      K::BoolArray(arr!([1, 1, 1u8])),
    ]))
  );
  assert_eq!(
    eval(&mut env, scan("3 3#2").unwrap()).unwrap(),
    Noun(K::List(vec![
      K::IntArray(arr!([2, 2, 2i64])),
      K::IntArray(arr!([2, 2, 2i64])),
      K::IntArray(arr!([2, 2, 2i64])),
    ]))
  );
  assert_eq!(
    eval(&mut env, scan("3 3#3.14").unwrap()).unwrap(),
    Noun(K::List(vec![
      K::FloatArray(arr!([3.14, 3.14, 3.14f64])),
      K::FloatArray(arr!([3.14, 3.14, 3.14f64])),
      K::FloatArray(arr!([3.14, 3.14, 3.14f64])),
    ]))
  );
  // assert_eq!(
  //   eval(&mut env, scan("3 3#\"a\"").unwrap()).unwrap(),
  //   Noun(K::List(vec![
  //     K::CharArray(arr!(['a', 'a', 'a'])),
  //     K::CharArray(arr!(['a', 'a', 'a'])),
  //     K::CharArray(arr!(['a', 'a', 'a'])),
  //   ]))
  // );
}

#[test]
fn test_split() {
  let mut env = Env { names: HashMap::new(), parent: None };

  let res1 = eval(&mut env, scan("\",\"\\\"lol,bang,biff\"").unwrap()).unwrap();
  println!("res1: {:?}", res1);
  assert_eq!(
    res1,
    Noun(K::List(vec![
      K::CharArray("lol".to_string()),
      K::CharArray("bang".to_string()),
      K::CharArray("biff".to_string())
    ]))
  );

  let res2 = eval(&mut env, scan("\"SPLIT\"\\\"lolSPLITbangSPLITbiff\"").unwrap()).unwrap();
  println!("res2: {:?}", res2);
  assert_eq!(
    res2,
    Noun(K::List(vec![
      K::CharArray("lol".to_string()),
      K::CharArray("bang".to_string()),
      K::CharArray("biff".to_string())
    ]))
  );
}

#[test]
fn test_join() {
  let mut env = Env { names: HashMap::new(), parent: None };

  let res1 = eval(&mut env, scan("\",\"/(\"lol\";\"bang\";\"biff\")").unwrap()).unwrap();
  println!("res1: {:?}", res1);
  assert_eq!(res1, Noun(K::CharArray("lol,bang,biff".to_string())));
}

#[test]
fn test_forced_monads() {
  let mut env = Env { names: HashMap::new(), parent: None };

  let res1 = eval(&mut env, scan("+:`a`b!(1 2 3;4 5 6)").unwrap()).unwrap();
  println!("res1: {:?}", res1);
  let res2 = eval(&mut env, scan("+`a`b!(1 2 3;4 5 6)").unwrap()).unwrap();
  assert_eq!(res1, res2);
}

#[test]
fn test_rand() {
  let mut env = Env { names: HashMap::new(), parent: None };

  let res1 = eval(&mut env, scan("5?5").unwrap()).unwrap();
  println!("res1: {:?}", res1);
  match res1 {
    KW::Noun(K::IntArray(s)) => {
      assert!(s.len() == 5);
      assert!(s.min::<i64>().unwrap().unwrap() >= 0);
      assert!(s.max::<i64>().unwrap().unwrap() < 5);
    }
    _ => panic!("wrong result"),
  }

  let res1 = eval(&mut env, scan("?5").unwrap()).unwrap();
  println!("res1: {:?}", res1);
  match res1 {
    KW::Noun(K::FloatArray(s)) => {
      assert!(s.len() == 5);
      assert!(s.min::<i64>().unwrap().unwrap() >= 0);
      assert!(s.max::<i64>().unwrap().unwrap() < 1);
    }
    _ => panic!("wrong result"),
  }

  let res1 = eval(&mut env, scan("5?1 2 3").unwrap()).unwrap();
  println!("res1: {:?}", res1);
  match res1 {
    KW::Noun(K::IntArray(s)) => {
      assert!(s.len() == 5);
      assert!(s.min::<i64>().unwrap().unwrap() >= 1);
      assert!(s.max::<i64>().unwrap().unwrap() <= 3);
    }
    _ => panic!("wrong result"),
  }
}

#[test]
fn test_find() {
  let mut env = Env { names: HashMap::new(), parent: None };

  let res1 = eval(&mut env, scan("\"abc\"?\"abcdef\"").unwrap()).unwrap();
  let res2 = eval(&mut env, scan("0 1 2 0N 0N 0N").unwrap()).unwrap();
  println!("res1: {:?}", res1);
  assert_eq!(res1, res2);
}

#[test]
fn test_each() {
  let mut env = Env { names: HashMap::new(), parent: None };

  let res1 = eval(&mut env, scan("!'3 3 3").unwrap()).unwrap();
  let res2 = eval(&mut env, scan("(0 1 2;0 1 2;0 1 2)").unwrap()).unwrap();
  println!("res1: {:?}", res1);
  assert_eq!(res1, res2);

  let res1 = eval(&mut env, scan("{2*x}'1 2 3").unwrap()).unwrap();
  let res2 = eval(&mut env, scan("2 4 6").unwrap()).unwrap();
  println!("res1: {:?}", res1);
  assert_eq!(res1, res2);
}

#[test]
fn test_eachright() {
  let mut env = Env { names: HashMap::new(), parent: None };

  let res1 = eval(&mut env, scan("a=/:a:!3").unwrap()).unwrap();
  let res2 = eval(&mut env, scan("(1 0 0;0 1 0;0 0 1)").unwrap()).unwrap();
  println!("res1: {:?}", res1);
  assert_eq!(res1, res2);
}

#[ignore]
#[test]
fn test_eachright_defun() {
  // TODO: parse adverbs properly
  let mut env = Env { names: HashMap::new(), parent: None };

  let res1 = eval(&mut env, scan("a{x=y}/:a:!3").unwrap()).unwrap();
  let res2 = eval(&mut env, scan("(1 0 0;0 1 0;0 0 1)").unwrap()).unwrap();
  println!("res1: {:?}", res1);
  assert_eq!(res1, res2);
}

#[test]
fn test_eachleft() {
  let mut env = Env { names: HashMap::new(), parent: None };

  let res1 = eval(&mut env, scan("3 4 5 +\\:(2 3;4 5;6)").unwrap()).unwrap();
  let res2 = eval(&mut env, scan("((5 6;7 8;9);(6 7;8 9;10);(7 8;9 10;11))").unwrap()).unwrap();
  println!("res1: {:?}", res1);
  assert_eq!(res1, res2);
}

#[test]
fn test_eachprior() {
  let mut env = Env { names: HashMap::new(), parent: None };

  let res1 = eval(&mut env, scan("-':1 6 2 3 4").unwrap()).unwrap();
  let res2 = eval(&mut env, scan("1 5 -4 1 1").unwrap()).unwrap();
  println!("res1: {:?}", res1);
  assert_eq!(res1, res2);
}

#[ignore]
#[test]
fn test_eachprior_d() {
  let mut env = Env { names: HashMap::new(), parent: None };

  let res1 = eval(&mut env, scan("5-':1 6 2 3 4").unwrap()).unwrap();
  let res2 = eval(&mut env, scan("-4 5 -4 1 1").unwrap()).unwrap();
  println!("res1: {:?}", res1);
  assert_eq!(res1, res2);
}

#[ignore]
#[test]
fn test_windows() {
  let mut env = Env { names: HashMap::new(), parent: None };

  let res1 = eval(&mut env, scan("3':1 2 3 4 5 6").unwrap()).unwrap();
  let res2 = eval(&mut env, scan("(1 2 3;2 3 4;3 4 5;4 5 6)").unwrap()).unwrap();
  println!("res1: {:?}", res1);
  assert_eq!(res1, res2);

  let res1 = eval(&mut env, scan("3 +/':1 2 3 4 5 6").unwrap()).unwrap();
  let res2 = eval(&mut env, scan("6 9 12 15").unwrap()).unwrap();
  println!("res1: {:?}", res1);
  assert_eq!(res1, res2);
}

#[test]
fn test_max() {
  let mut env = Env { names: HashMap::new(), parent: None };

  assert_eq!(eval(&mut env, scan("1|2").unwrap()).unwrap(), Noun(K::Int(Some(2))));
  assert_eq!(
    eval(&mut env, scan("1|1 2 3").unwrap()).unwrap(),
    Noun(K::IntArray(arr!([1, 2, 3i64])))
  );
  assert_eq!(
    eval(&mut env, scan("1 2 3|1").unwrap()).unwrap(),
    Noun(K::IntArray(arr!([1, 2, 3i64])))
  );

  assert_eq!(
    eval(&mut env, scan("1.0|1 2 3").unwrap()).unwrap(),
    Noun(K::FloatArray(arr!([1.0, 2.0, 3.0f64])))
  );
  assert_eq!(
    eval(&mut env, scan("1 2 3|1.0").unwrap()).unwrap(),
    Noun(K::FloatArray(arr!([1.0, 2.0, 3.0f64])))
  );
}

#[test]
fn test_min() {
  let mut env = Env { names: HashMap::new(), parent: None };

  assert_eq!(eval(&mut env, scan("1&2").unwrap()).unwrap(), Noun(K::Int(Some(1))));
  assert_eq!(
    eval(&mut env, scan("1&1 2 3").unwrap()).unwrap(),
    Noun(K::IntArray(arr!([1, 1, 1i64])))
  );
  assert_eq!(
    eval(&mut env, scan("1 2 3&1").unwrap()).unwrap(),
    Noun(K::IntArray(arr!([1, 1, 1i64])))
  );

  assert_eq!(
    eval(&mut env, scan("1.0&1 2 3").unwrap()).unwrap(),
    Noun(K::FloatArray(arr!([1.0, 1.0, 1.0f64])))
  );
  assert_eq!(
    eval(&mut env, scan("1 2 3&1.0").unwrap()).unwrap(),
    Noun(K::FloatArray(arr!([1.0, 1.0, 1.0f64])))
  );
}

#[test]
fn test_promote_nouns() {
  let mut env = Env { names: HashMap::new(), parent: None };

  println!("promote_nouns(1, 1 2 3)");
  let l = eval(&mut env, scan("1").unwrap()).unwrap().unwrap_noun();
  let r = eval(&mut env, scan("1 2 3").unwrap()).unwrap().unwrap_noun();
  assert_eq!(promote_nouns(l, r.clone()), (K::IntArray(arr!([1, 1, 1i64])), r));

  println!("promote_nouns(1 2 3, 1)");
  let l = eval(&mut env, scan("1 2 3").unwrap()).unwrap().unwrap_noun();
  let r = eval(&mut env, scan("1").unwrap()).unwrap().unwrap_noun();
  assert_eq!(promote_nouns(l.clone(), r), (l, K::IntArray(arr!([1, 1, 1i64]))));

  println!("promote_nouns(1, 1 2 3.0)");
  let l = eval(&mut env, scan("1").unwrap()).unwrap().unwrap_noun();
  let r = eval(&mut env, scan("1 2 3.0").unwrap()).unwrap().unwrap_noun();
  assert_eq!(promote_nouns(l, r.clone()), (K::FloatArray(arr!([1.0, 1.0, 1.0f64])), r));

  println!("promote_nouns(1 2 3.0, 1)");
  let l = eval(&mut env, scan("1 2 3.0").unwrap()).unwrap().unwrap_noun();
  let r = eval(&mut env, scan("1").unwrap()).unwrap().unwrap_noun();
  assert_eq!(promote_nouns(l.clone(), r), (l, K::FloatArray(arr!([1.0, 1.0, 1.0f64]))));
}

#[ignore]
#[test]
fn test_split_strings() {
  let mut env = Env { names: HashMap::new(), parent: None };

  //TODO fix parse error of 2 or more adverb chain
  let res = eval(&mut env, scan(r#"","\'("1,2";"3,4")"#).unwrap()).unwrap().unwrap_noun();

  assert_eq!(
    res,
    K::List(vec![
      K::List(vec![K::Char('1'), K::Char('2')]),
      K::List(vec![K::Char('3'), K::Char('4')])
    ])
  );
}

#[test]
fn test_eval_verb() {
  let mut env = Env { names: HashMap::new(), parent: None };

  let res = eval(&mut env, scan(r#"."42""#).unwrap()).unwrap().unwrap_noun();

  assert_eq!(res, K::Int(Some(42)));
}

#[test]
fn test_concat() {
  let mut env = Env { names: HashMap::new(), parent: None };

  let res = eval(&mut env, scan("2,3").unwrap()).unwrap().unwrap_noun();
  assert_eq!(res, K::IntArray(arr!([2, 3i64])));

  let res = eval(&mut env, scan("1 2,3 4.0").unwrap()).unwrap().unwrap_noun();
  assert_eq!(res, K::FloatArray(arr!([1., 2., 3., 4.0f64])));
}

#[test]
fn test_grade() {
  let mut env = Env { names: HashMap::new(), parent: None };

  let res = eval(&mut env, scan("< 3 2 1").unwrap()).unwrap().unwrap_noun();
  assert_eq!(res, K::IntArray(arr!([2, 1, 0i64])));

  let res = eval(&mut env, scan("< 3.0 2.5 1").unwrap()).unwrap().unwrap_noun();
  assert_eq!(res, K::IntArray(arr!([2, 1, 0i64])));

  let res = eval(&mut env, scan("> 3 2 1").unwrap()).unwrap().unwrap_noun();
  assert_eq!(res, K::IntArray(arr!([0i64, 1, 2])));
}

#[test]
fn test_index_take_drop_bounds() {
  assert_eq!(k_eval("1 _ 1 2 3"), k_eval("2 3"));

  assert_eq!(k_eval("1 2 3 @ 42"), k_eval("0N"));

  assert_eq!(k_eval("1 2 3 @ 0 1 42"), k_eval("1 2 0N"));

  assert_eq!(k_eval("0 # 1 2 3"), k_eval("!0"));

  assert_eq!(k_eval("0N # 1 2 3"), k_eval("1 2 3"));
}

#[test]
fn test_group() {
  assert_eq!(k_eval("= 1 1 2 2 3 3 4"), k_eval("`1`2`3`4!(0 1;2 3;4 5;(,6))"));
  assert_eq!(k_eval("= 1.0 1 2 2 3 3 4"), k_eval("`1.0`2.0`3.0`4.0!(0 1;2 3;4 5;(,6))"));

  assert_eq!(k_eval("= \"foo\""), k_eval("`f`o!((,0);1 2)"));

  // TODO
  // assert_eq!(k_eval("= (0 1;1 0;0 1)"), k_eval("!/+((0 1;0 2);(1 0;,1))"));
  assert_eq!(k_eval("= (0 1;1 0;0 1)"), k_eval("`01b`10b!(0 2;(,1))"));
}

#[test]
fn test_imat() {
  assert_eq!(k_eval("=3"), k_eval("(1 0 0;0 1 0;0 0 1)"));
}

#[test]
fn test_reverse() {
  assert_eq!(k_eval("|1"), k_eval("1"));
  assert_eq!(k_eval("|2"), k_eval("2"));
  assert_eq!(k_eval("|1 2 3"), k_eval("3 2 1"));
  assert_eq!(k_eval("|1 2 3.0"), k_eval("3.0 2 1"));
  assert_eq!(k_eval("|\"abc\""), k_eval("\"cba\""));
  assert_eq!(k_evals("|`a`b`c"), k_evals("`c`b`a"));
  assert_eq!(k_eval("|`a`b`c!(1;2;3)"), k_eval("`c`b`a!(3;2;1)"));
  assert_eq!(k_eval("|+`a`b`c!(1 2 3;4 5 6;\"abc\")"), k_eval("+`a`b`c!(3 2 1;6 5 4;\"cba\")"));
  assert_eq!(k_eval("|(`a;1;\"b\")"), k_eval("(\"b\";1;`a)"));
}

#[test]
fn test_comparisons() {
  println!("test_comparisons() numbers");
  assert_eq!(k_eval("1<2"), k_eval("1"));
  assert_eq!(k_eval("1<1 2 3"), k_eval("0 1 1"));
  assert_eq!(k_eval("1.0<2"), k_eval("1"));
  assert_eq!(k_eval("0N<1"), k_eval("1"));
  assert_eq!(k_eval("0N<0N"), k_eval("0"));

  println!("test_comparisons() dicts");
  assert_eq!(k_eval("1<`a`b!(1;2)"), k_eval("`a`b!(0;1)"));
  assert_eq!(k_eval("1<`a`b!(1;2 3 4)"), k_eval("`a`b!(0;1 1 1)"));
  assert_eq!(k_eval("(`a`b!(1;2))<2"), k_eval("`a`b!(1;0)"));
  assert_eq!(k_eval("(`a`b!(1;2 3 4))<3"), k_eval("`a`b!(1;1 0 0)"));
  assert_eq!(k_eval("(`a`b!(1;1))<(`a`b`c!(1;2;3 4 5))"), k_eval("`a`b`c!(0;1;1 1 1)"));
  assert_eq!(k_eval("(`a`b!(1;1))<(`a`b`c!(1;2;0N))"), k_eval("`a`b`c!(0;1;0)"));

  println!("test_comparisons() tables");
  assert_eq!(k_eval("1<+`a`b!(1;2 3 4)"), k_eval("+`a`b!(0 0 0;1 1 1)"));
  assert_eq!(k_eval("(+`a`b!(1;2 3 4))<3"), k_eval("+`a`b!(1 1 1;1 0 0)"));

  assert_eq!(k_eval("1>2"), k_eval("0"));
}
