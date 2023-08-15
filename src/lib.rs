use itertools::Itertools;
use log::debug;
use polars::prelude::*;
use std::{collections::VecDeque, iter::repeat, ops};

// oK.js is 1k lines of javascript in one file for a k6 interpreter.
// The Challenge: Can we do the same in RUST?
// (Probably gonna need a lot of macros and to turn of cargo fmt maybe?)
//

// Initially I thought all K Arrays should be arrow arrays...
// but what if they were polars.Series and polars.DataFrames!?
// Then we'd get fast csv/parquet/json for "free".

#[derive(Clone, Debug, PartialEq)]
pub enum K {
  Bool(u8),
  Int(Option<i64>), //blech option here
  Float(f64),
  Char(char),
  Symbol(String),
  SymbolArray(Series),
  BoolArray(Series),
  IntArray(Series),
  FloatArray(Series),
  CharArray(Series),
  Nil, // Is Nil a noun?
  List(Vec<K>),
  Dictionary(Box<K>, Box<K>), // Dictionary(SymbolArray, List)
                              //Table{ DataFrame },
                              //Quote(Box<K>) // Is Quote a noun?
}
#[derive(Clone, Debug, PartialEq)]
pub enum KW /* KWords */ {
  Noun(K),
  // Function{ body, args, curry, env }
  // View{ value, r, cache, depends->val }
  // NameRef { name, l(index?), r(assignment), global? }
  // Verb { name: String, l: Option<Box<K>>, r: Box<K>, curry: Option<Vec<K>>, },
  Verb { name: String },
  // Adverb { name, l(?), verb, r }
  // Cond { body: Vec< Vec<K> > } //list of expressions...
  StartOfLine,
  Nothing,
  LP,
  RP,
  SC, // semicolon
}

impl K {
  pub fn len<'s>(&'s self) -> usize {
    use K::*;
    match self {
      Nil => 0,
      BoolArray(a) => a.len(),
      IntArray(a) => a.len(),
      FloatArray(a) => a.len(),
      CharArray(a) => a.len(),
      _ => 1,
    }
  }
}

#[macro_export]
macro_rules! arr {
  ($v:expr) => {
    Series::new("", $v)
  };
}

pub fn vec2list(nouns: Vec<KW>) -> Result<K, &'static str> {
  //TODO handle same atom types promotion: (1;2;3) => IntArray([1 2 3])
  if nouns.iter().all(|w| matches!(w, KW::Noun(_))) {
    // check they're all nouns and make a List of the K objects within
    let v = nouns
      .iter()
      .map(|w| match w {
        KW::Noun(n) => n.clone(),
        _ => K::Nil,
      })
      .collect();
    Ok(K::List(v))
  } else {
    Err("invalid list")
  }
}

pub fn apply_primitive(v: &str, l: Option<KW>, r: KW) -> Result<KW, &'static str> {
  match v {
    "+" => match (l, r) {
      (Some(KW::Noun(l)), KW::Noun(r)) => v_plus(l, r).and_then(|n| Ok(KW::Noun(n))),
      _ => todo!("monad +"),
    },
    "-" => match (l, r) {
      (Some(KW::Noun(l)), KW::Noun(r)) => v_minus(l, r).and_then(|n| Ok(KW::Noun(n))),
      _ => todo!("monad -"),
    },
    "*" => match (l, r) {
      (Some(KW::Noun(l)), KW::Noun(r)) => v_times(l, r).and_then(|n| Ok(KW::Noun(n))),
      _ => todo!("monad *"),
    },
    "%" => match (l, r) {
      (Some(KW::Noun(l)), KW::Noun(r)) => v_divide(l, r).and_then(|n| Ok(KW::Noun(n))),
      _ => todo!("monad %"),
    },
    "!" => match (l, r) {
      (None, KW::Noun(r)) => Ok(KW::Noun(v_bang(r).unwrap())),
      (Some(KW::Noun(l)), KW::Noun(r)) => Ok(KW::Noun(v_d_bang(l, r).unwrap())),
      _ => todo!("wat"),
    },
    _ => Err("invalid primitive"),
  }
}

pub fn b2i(b: K) -> K {
  match b {
    K::BoolArray(b) => K::IntArray(
      b.bool()
        .expect("bool")
        .into_iter()
        .map(|b| match b {
          Some(true) => Some(1),
          Some(false) => Some(0),
          _ => None,
        })
        .collect::<Series>(),
    ),
    _ => panic!("not bool"),
  }
}

macro_rules! impl_op {
    ($op:tt, $opf:ident, $self:ident, $r:ident) => {
        match ($self.clone(), $r) {
            (K::Bool(l), K::Bool(r)) => K::Int(Some(l as i64 $op r as i64)),
            (K::Bool(l), K::Int(Some(r))) => K::Int(Some(l as i64 $op r)),
            (K::Bool(l), K::Float(r)) => K::Float(l as f64 $op r),
            (K::Bool(l), K::BoolArray(r)) => K::IntArray(polars::prelude::LhsNumOps::$opf(l as i64, &r)),
            (K::Bool(l), K::IntArray(r)) => K::IntArray(polars::prelude::LhsNumOps::$opf(l as i64, &r)),
            (K::Bool(l), K::FloatArray(r)) => K::FloatArray(polars::prelude::LhsNumOps::$opf(l as f64, &r)),
            (K::Int(Some(l)), K::Bool(r)) => K::Int(Some(l as i64 $op r as i64)),
            (K::Int(Some(l)), K::Int(Some(r))) => K::Int(Some(l as i64 $op r)),
            (K::Int(Some(l)), K::Float(r)) => K::Float(l as f64 $op r),
            (K::Int(Some(l)), K::BoolArray(r)) => K::IntArray(polars::prelude::LhsNumOps::$opf(l as i64, &r)),
            (K::Int(Some(l)), K::IntArray(r)) => K::IntArray(polars::prelude::LhsNumOps::$opf(l as i64, &r)),
            (K::Int(Some(l)), K::FloatArray(r)) => K::FloatArray(polars::prelude::LhsNumOps::$opf(l as f64, &r)),
            (K::Int(None), K::Bool(_)) => K::Int(None),
            (K::Int(None), K::Int(Some(_))) => K::Int(None),
            (K::Int(None), K::Float(_)) => K::Float(f64::NAN),
            (K::Int(None), K::BoolArray(r)) => K::IntArray(Series::new_null("",r.len())), // not sure about this
            (K::Int(None), K::IntArray(r)) => K::IntArray(Series::new_null("",r.len())),
            (K::Int(None), K::FloatArray(r)) => K::FloatArray(Series::new_null("",r.len())),
            (K::Float(l), K::Bool(r)) => K::Float(l $op r as f64),
            (K::Float(l), K::Int(Some(r))) => K::Float(l $op r as f64),
            (K::Float(l), K::Float(r)) => K::Float(l $op r),
            (K::Float(l), K::BoolArray(r)) => K::FloatArray(polars::prelude::LhsNumOps::$opf(l, &r)),
            (K::Float(l), K::IntArray(r)) => K::FloatArray(polars::prelude::LhsNumOps::$opf(l, &r.cast(&DataType::Float64).unwrap())),
            (K::Float(l), K::FloatArray(r)) => K::FloatArray(polars::prelude::LhsNumOps::$opf(l, &r)),
            (K::BoolArray(l), K::Bool(r)) => K::IntArray(l $op r),
            (K::BoolArray(_l), K::Int(Some(r))) => match b2i($self) { K::IntArray(l) => K::IntArray(l $op r), _ => panic!("impossible"), },
            (K::BoolArray(l), K::Float(r)) => K::FloatArray(l $op r),
            (K::BoolArray(l), K::BoolArray(r)) => K::IntArray(l $op r),
            (K::BoolArray(l), K::IntArray(r)) => K::IntArray(l $op r),
            (K::BoolArray(l), K::FloatArray(r)) => K::FloatArray(l $op r),
            (K::IntArray(l), K::Bool(r)) => K::IntArray(l $op r),
            (K::IntArray(l), K::Int(Some(r))) => K::IntArray(l $op r),
            (K::IntArray(l), K::Float(r)) => K::FloatArray(l $op r),
            (K::IntArray(l), K::BoolArray(r)) => K::IntArray(l $op r),
            (K::IntArray(l), K::IntArray(r)) => K::IntArray(l $op r),
            (K::IntArray(l), K::FloatArray(r)) => K::FloatArray(l $op r),
            (K::FloatArray(l), K::Bool(r)) => K::FloatArray(l $op r as f64),
            (K::FloatArray(l), K::Int(Some(r))) => K::FloatArray(l $op r as f64),
            (K::FloatArray(l), K::Float(r)) => K::FloatArray(l $op r),
            (K::FloatArray(l), K::BoolArray(r)) => K::FloatArray(l $op r),
            (K::FloatArray(l), K::IntArray(r)) => K::FloatArray(l $op r),
            (K::FloatArray(l), K::FloatArray(r)) => K::FloatArray(l $op r),
            _ => todo!("various $op pairs - LOTS MORE to do still: char/dicts/tables/etc"),
        }
    };
}

impl ops::Add for K {
  type Output = Self;
  fn add(self, r: Self) -> Self::Output { impl_op!(+, add, self, r) }
}
impl ops::Sub for K {
  type Output = Self;
  fn sub(self, r: Self) -> Self::Output { impl_op!(-, sub, self, r) }
}
impl ops::Mul for K {
  type Output = Self;
  fn mul(self, r: Self) -> Self::Output { impl_op!(*, mul, self, r) }
}
impl ops::Div for K {
  type Output = Self;
  fn div(self, r: Self) -> Self::Output { impl_op!(/, div, self, r) }
}

fn len_ok(l: &K, r: &K) -> Result<bool, &'static str> {
  if l.len() == r.len() || l.len() == 1 || r.len() == 1 {
    Ok(true)
  } else {
    Err("length")
  }
}
pub fn v_plus(l: K, r: K) -> Result<K, &'static str> { len_ok(&l, &r).and_then(|_| Ok(l + r)) }
pub fn v_minus(l: K, r: K) -> Result<K, &'static str> { len_ok(&l, &r).and_then(|_| Ok(l - r)) }
pub fn v_times(l: K, r: K) -> Result<K, &'static str> { len_ok(&l, &r).and_then(|_| Ok(l * r)) }
pub fn v_divide(l: K, r: K) -> Result<K, &'static str> { len_ok(&l, &r).and_then(|_| Ok(l / r)) }
pub fn v_bang(r: K) -> Result<K, &'static str> {
  debug!("v_bang");
  match r {
    K::Int(Some(i)) => Ok(K::IntArray(arr![(0..i).collect::<Vec<i64>>()])),
    _ => todo!("v_bang variants"),
  }
}
pub fn v_d_bang(l: K, r: K) -> Result<K, &'static str> {
  match (l, r) {
    (K::SymbolArray(s), K::List(v)) => {
      if s.len() == v.len() {
        Ok(K::Dictionary(Box::new(K::SymbolArray(s)), Box::new(K::List(v))))
      } else if v.len() == 1 {
        Ok(K::Dictionary(
          Box::new(K::SymbolArray(s.clone())),
          Box::new(K::List(std::iter::repeat(v[0].clone()).take(s.len()).collect())),
        ))
      } else {
        Err("length")
      }
    }
    _ => {
      todo!("modulo")
      // len_ok(&l, &r).and_then(|_| Ok(l % r))
    }
  }
}

pub fn eval(sentence: Vec<KW>) -> Result<KW, &'static str> {
  let mut queue = VecDeque::from([vec![KW::StartOfLine], sentence].concat());
  let mut stack: VecDeque<KW> = VecDeque::new();

  let mut converged: bool = false;
  while !converged {
    // debug!("stack: {stack:?}");
    // let fragment: Vec<KW> = stack.drain(..4).collect();
    let fragment = get_fragment(&mut stack);
    let result: Result<Vec<KW>, &'static str> = match fragment {
      (w, KW::Verb { name }, x @ KW::Noun(_), any) if matches!(w, KW::StartOfLine | KW::LP) => {
        // 0 monad
        apply_primitive(&name, None, x.clone()).and_then(|r| Ok(vec![w, r, any]))
      }
      (w, v @ KW::Verb { .. }, KW::Verb { name }, x @ KW::Noun(_)) => {
        // 1 monad
        apply_primitive(&name, None, x.clone()).and_then(|r| Ok(vec![w, v, r]))
      }
      (any, x @ KW::Noun(_), KW::Verb { name }, y @ KW::Noun(_)) => {
        // 2 dyad
        apply_primitive(&name, Some(x.clone()), y.clone()).and_then(|r| Ok(vec![any, r]))
      }
      // TODO: rest of the J (K is similar!) parse table (minus forks/hooks) https://www.jsoftware.com/help/jforc/parsing_and_execution_ii.htm#_Toc191734587
      (KW::LP, w, KW::RP, any) => Ok(vec![w.clone(), any.clone()]), // 8 paren
      (KW::LP, KW::Noun(n1), KW::SC, KW::Noun(n2)) => {
        // List
        if let Some(i) = stack.iter().position(|w| matches!(w, KW::RP)) {
          // Pull off stack until first KW::RP, Drop all KW::SC and KW::RP tokens.
          let nouns: VecDeque<KW> =
            stack.drain(..i + 1).filter(|w| !matches!(*w, KW::SC) && !matches!(*w, KW::RP)).collect();
          Ok(vec![KW::Noun(
            vec2list([vec![KW::Noun(n1), KW::Noun(n2)], nouns.into()].concat()).unwrap(),
          )])
        } else {
          Err("invalid list syntax")
        }
      }
      (w1, w2, w3, w4) => match queue.pop_back() {
        Some(v) => Ok(vec![v, w1.clone(), w2.clone(), w3.clone(), w4.clone()]),
        None => {
          converged = true;
          Ok(vec![w1.clone(), w2.clone(), w3.clone(), w4.clone()])
        }
      },
    };

    stack.retain(|w| !matches!(w, KW::Nothing));
    debug!("result: {:?} with stack: {:?}", result, stack);
    stack = [result?, stack.into()].concat().into();
  }
  stack.retain(|w| !matches!(w, KW::StartOfLine | KW::Nothing));
  let r: Vec<KW> = stack.iter().cloned().collect();
  if r.len() == 1 {
    Ok(r[0].clone())
  } else {
    debug!("{:?}", r);
    Err("invalid result stack")
  }
}

fn get_fragment(stack: &mut VecDeque<KW>) -> (KW, KW, KW, KW) {
  stack
    .drain(..stack.len().min(4))
    .chain(repeat(KW::Nothing))
    .next_tuple()
    .expect("infinite iterator can't be empty")
}

pub fn scan(code: &str) -> Result<Vec<KW>, &'static str> {
  let mut words = vec![];
  let mut skip: usize = 0;
  for (i, c) in code.char_indices() {
    if skip > 0 {
      skip -= 1;
      continue;
    }
    match c {
      '(' => words.push(KW::LP),
      ')' => words.push(KW::RP),
      ';' => words.push(KW::SC),
      '0'..='9' | '-' => {
        if let Ok((j, k)) = scan_number(&code[i..]) {
          words.push(k);
          skip = j;
        } else {
          words.push(KW::Verb { name: c.to_string() })
        }
      }
      '"' => {
        let (j, k) = scan_string(&code[i..]).unwrap();
        words.push(k);
        skip = j;
      }
      '`' => {
        let (j, k) = scan_symbol(&code[i..]).unwrap();
        words.push(k);
        skip = j;
      }
      ':' | '+' | '*' | '%' | '!' | '&' | '|' | '<' | '>' | '=' | '~' | ',' | '^' | '#' | '_'
      | '$' | '?' | '@' | '.' => words.push(KW::Verb { name: c.to_string() }),
      ' ' | '\t' | '\n' => continue,
      _ => return Err("TODO"),
    };
  }
  Ok(words)
}

pub fn scan_number(code: &str) -> Result<(usize, KW), &'static str> {
  // read until first char outside 0123456789.-
  // split on space and parse to numeric
  //
  // an array *potentially* extends until the first symbol character
  let sentence = match code.find(|c: char| {
    !(c.is_ascii_alphanumeric() || c.is_ascii_whitespace() || ['.', '-', 'n', 'N'].contains(&c))
  }) {
    Some(c) => &code[..c],
    None => code,
  };

  // split on the whitespace, and try to parse each 'word', stopping when we can't parse a word
  let parts: Vec<(&str, K)> = sentence
    .split_whitespace()
    .map_while(|term| scan_num_token(term).ok().map(|x| (term, x)))
    .collect();

  // the end is the end of the last successfully parsed term
  if let Some((term, _)) = parts.last() {
    let l = term.as_ptr() as usize - sentence.as_ptr() as usize + term.len() - 1;

    let nums: Vec<K> = parts.into_iter().map(|(_term, num)| num).collect();
    match nums.len() {
      0 => panic!("impossible"),
      1 => Ok((l, KW::Noun(nums[0].clone()))),
      _ => Ok((l, KW::Noun(promote_num(nums).unwrap()))),
    }
  } else {
    Err("syntax error: a sentence starting with a digit must contain a valid number")
  }
}

pub fn scan_string(code: &str) -> Result<(usize, KW), &'static str> {
  // read string, accounting for C style escapes: \", \n, \t, etc
  if code.chars().nth(0) != Some('"') {
    panic!("called scan_string() on invalid input")
  } else {
    let mut i: usize = 1;
    let mut s = String::new();
    // TODO: Yeah this is awful...
    while i < code.len() {
      if code.chars().nth(i) == Some('"') {
        return Ok(match s.len() {
          // Does k really have char atoms?
          1 => (i, KW::Noun(K::Char(s.chars().nth(0).unwrap()))),
          _ => (i, KW::Noun(K::CharArray(Series::new("", &s)))),
        });
      } else if code.chars().nth(i) == Some('\\') {
        match code.chars().nth(i + 1) {
          Some('\\') => s.push_str("\\"),
          Some('t') => s.push_str("\t"),
          Some('n') => s.push_str("\n"),
          // TODO: handle the rest
          _ => return Err("parse error: invalid string"),
        }
        i += 1; // skip next char.
      } else {
        s.extend(code.chars().nth(i));
      }
      i += 1;
    }
    Err("parse error: unmatched \"")
  }
}

pub fn scan_symbol(code: &str) -> Result<(usize, KW), &'static str> {
  // read a Symbol or SymbolArray
  //
  // read until first char outside [a-z0-9`]
  // split on ` or space`
  //
  // a SymbolArray *potentially* extends until the first symbol character
  let sentence = match code
    .find(|c: char| !(c.is_ascii_alphanumeric() || c.is_ascii_whitespace() || ['`'].contains(&c)))
  {
    Some(c) => &code[..c],
    None => code,
  };

  if sentence.chars().nth(0) != Some('`') {
    panic!("called scan_symbol() on invalid input")
  } else {
    let mut i: usize = 1;
    let mut s = String::new();
    let mut ss: Vec<String> = vec![];
    let mut b_in_sym = true;
    // TODO: Yeah this is awful...
    while i < sentence.len() {
      if sentence.chars().nth(i) == Some(' ') {
        if b_in_sym {
          ss.extend(vec![s.clone()]);
          s.clear();
        }
        b_in_sym = false;
      } else if sentence.chars().nth(i) == Some('`') {
        if b_in_sym {
          ss.extend(vec![s.clone()]);
          s.clear();
        }
        b_in_sym = true;
      } else {
        if b_in_sym {
          s.extend(sentence.chars().nth(i));
        } else {
          break;
        }
      }
      i += 1;
    }
    if s.len() > 0 || sentence.chars().nth(sentence.len() - 1) == Some('`') {
      // catch trailing empty symbol eg: `a`b`c` (SymbolArray(["a","b","c",""]))
      ss.extend(vec![s.clone()]);
    }
    match ss.len() {
      0 => panic!("wat - invalid scansymbol()"),
      // 0 => return Ok((i, KW::Noun(K::Symbol(s)))),
      1 => return Ok((i, KW::Noun(K::Symbol(ss[0].clone())))),
      _ => {
        return Ok((
          i,
          KW::Noun(K::SymbolArray(
            Series::new("a", ss).cast(&DataType::Categorical(None)).unwrap(),
          )),
        ))
      }
    }
  }
}

pub fn scan_num_token(term: &str) -> Result<K, &'static str> {
  match term {
    "0N" => Ok(K::Int(None)),
    "0n" => Ok(K::Float(f64::NAN)),
    _ => {
      if let Ok(i) = term.parse::<u8>() {
        match i {
          0 | 1 => Ok(K::Bool(i)),
          _ => Ok(K::Int(Some(i as i64))),
        }
      } else if let Ok(i) = term.parse::<i64>() {
        Ok(K::Int(Some(i)))
      } else if let Ok(f) = term.parse::<f64>() {
        Ok(K::Float(f))
      } else {
        Err("invalid num token")
      }
    }
  }
}

pub fn promote_num(nums: Vec<K>) -> Result<K, &'static str> {
  if nums.iter().any(|k| matches!(k, K::Float(_))) {
    let fa: Vec<f64> = nums
      .iter()
      .map(|k| match k {
        K::Bool(i) => *i as f64,
        K::Int(None) => f64::NAN,
        K::Int(Some(i)) => *i as f64,
        K::Float(f) => *f,
        _ => panic!("invalid float"),
      })
      .collect();

    Ok(K::FloatArray(Series::new("", fa)))
  } else if nums.iter().any(|k| matches!(k, K::Int(_))) {
    let ia: Vec<Option<i64>> = nums
      .iter()
      .map(|k| match k {
        K::Bool(i) => Some(*i as i64),
        K::Int(i) => *i,
        _ => panic!("invalid int"),
      })
      .collect();

    Ok(K::IntArray(Series::new("", ia)))
  } else if nums.iter().all(|k| matches!(k, K::Bool(_))) {
    let ba: BooleanChunked = nums
      .iter()
      .map(|k| match k {
        K::Bool(0) => false,
        K::Bool(1) => true,
        _ => panic!("invalid bool"),
      })
      .collect();

    Ok(K::BoolArray(Series::new("", ba)))
  } else {
    Err("invalid nums")
  }
}
