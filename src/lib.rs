use itertools::Itertools;
use log::debug;
use polars::prelude::*;
use indexmap::IndexMap;
use std::fs::File;
use std::iter::zip;
use std::path::Path;
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
  Dictionary(IndexMap<String, K>),
  Table(DataFrame),
  //Quote(Box<K>) // Is Quote a noun?
}
#[derive(Clone, Debug, PartialEq)]
pub enum KW /* KWords */ {
  Noun(K),
  // Function{ body, args, curry, env }
  // View{ value, r, cache, depends->val }
  Name(String),
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
      Dictionary(d) => d.len(),
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

// pub fn enlist(k: K) -> Result<K, &'static str> {
pub fn k_to_vec(k: K) -> Result<Vec<K>, &'static str> {
  match k {
    K::List(v) => Ok(v),
    // TODO: reduce this repetition with a macro
    K::BoolArray(a) => Ok(
      a.iter()
        .map(|i| {
          return if i.try_extract::<u8>().is_ok() {
            K::Bool(i.try_extract::<u8>().unwrap())
          } else {
            panic!("oops")
          };
        })
        .collect(),
    ),
    K::IntArray(v) => Ok(
      v.iter()
        .map(|i| {
          return if i.try_extract::<i64>().is_ok() {
            K::Int(Some(i.try_extract::<i64>().unwrap()))
          } else if i.is_nested_null() {
            K::Int(None)
          } else {
            panic!("oops")
          };
        })
        .collect(),
    ),
    K::FloatArray(v) => Ok(
      v.iter()
        .map(|i| {
          return if i.try_extract::<f64>().is_ok() {
            K::Float(i.try_extract::<f64>().unwrap())
          } else if i.is_nested_null() {
            K::Float(f64::NAN)
          } else {
            panic!("oops")
          };
        })
        .collect(),
    ),
    K::CharArray(v) => {
      Ok(v.u8().unwrap().into_iter().map(|c| K::Char(c.unwrap() as char)).collect())
    }
    K::SymbolArray(_v) => {
      todo!("enlist(SymbolArray(...))")
    }
    _ => Err("todo"),
  }
}
pub fn vec2list(nouns: Vec<KW>) -> Result<K, &'static str> {
  if nouns.iter().all(|w| matches!(w, KW::Noun(K::Bool(_)))) {
    let v: Vec<u8> = nouns
      .iter()
      .map(|w| match w {
        KW::Noun(K::Bool(b)) => *b,
        _ => panic!("impossible"),
      })
      .collect();
    Ok(K::BoolArray(arr!(v)))
  } else if nouns
    .iter()
    .all(|w| matches!(w, KW::Noun(K::Bool(_))) || matches!(w, KW::Noun(K::Int(Some(_)))))
  {
    let v: Vec<i64> = nouns
      .iter()
      .map(|w| match w {
        KW::Noun(K::Bool(b)) => *b as i64,
        KW::Noun(K::Int(Some(i))) => *i,
        _ => panic!("impossible"),
      })
      .collect();
    Ok(K::IntArray(arr!(v)))
  } else if nouns
    .iter()
    .all(|w| matches!(w, KW::Noun(K::Bool(_))) || matches!(w, KW::Noun(K::Float(_))))
  {
    let v: Vec<f64> = nouns
      .iter()
      .map(|w| match w {
        KW::Noun(K::Bool(b)) => *b as f64,
        KW::Noun(K::Float(f)) => *f,
        _ => panic!("impossible"),
      })
      .collect();
    Ok(K::FloatArray(arr!(v)))
  } else if nouns.iter().all(|w| matches!(w, KW::Noun(K::Char(_)))) {
    let v: String = nouns
      .iter()
      .map(|w| match w {
        KW::Noun(K::Char(c)) => *c,
        _ => panic!("impossible"),
      })
      .collect();
    Ok(K::CharArray(arr!(v)))
  } else if nouns.iter().all(|w| matches!(w, KW::Noun(_))) {
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

type V1 = fn(K) -> Result<K, &'static str>;
type V2 = fn(K, K) -> Result<K, &'static str>;
type V3 = fn(K, K, K) -> Result<K, &'static str>;
type V4 = fn(K, K, K, K) -> Result<K, &'static str>;

pub fn v_none1(_x: K) -> Result<K, &'static str> { Err("rank") }
pub fn v_none2(_x: K, _y: K) -> Result<K, &'static str> { Err("rank") }
pub fn v_none3(_x: K, _y: K, _z: K) -> Result<K, &'static str> { Err("rank") }
pub fn v_none4(_a: K, _b: K, _c: K, _d: K) -> Result<K, &'static str> { Err("rank") }

pub fn apply_primitive(v: &str, l: Option<KW>, r: KW) -> Result<KW, &'static str> {
  // See https://github.com/JohnEarnest/ok/blob/gh-pages/oK.js
  //        a          l           a-a         l-a         a-l         l-l         triad    tetrad
  // ":" : [ident,     ident,      rident,     rident,     rident,     rident,     null,    null  ],
  // "+" : [flip,      flip,       ad(plus),   ad(plus),   ad(plus),   ad(plus),   null,    null  ],
  // "-" : [am(negate),am(negate), ad(minus),  ad(minus),  ad(minus),  ad(minus),  null,    null  ],
  // "*" : [first,     first,      ad(times),  ad(times),  ad(times),  ad(times),  null,    null  ],
  // "%" : [am(sqrt),  am(sqrt),   ad(divide), ad(divide), ad(divide), ad(divide), null,    null  ],
  // "!" : [iota,      odometer,   mod,        null,       ar(mod),    md,         null,    null  ],
  // "&" : [where,     where,      ad(min),    ad(min),    ad(min),    ad(min),    null,    null  ],
  // "|" : [rev,       rev,        ad(max),    ad(max),    ad(max),    ad(max),    null,    null  ],
  // "<" : [asc,       asc,        ad(less),   ad(less),   ad(less),   ad(less),   null,    null  ],
  // ">" : [desc,      desc,       ad(more),   ad(more),   ad(more),   ad(more),   null,    null  ],
  // "=" : [imat,      group,      ad(equal),  ad(equal),  ad(equal),  ad(equal),  null,    null  ],
  // "~" : [am(not),   am(not),    match,      match,      match,      match,      null,    null  ],
  // "," : [enlist,    enlist,     cat,        cat,        cat,        cat,        null,    null  ],
  // "^" : [pisnull,   am(pisnull),ad(fill),   except,     ad(fill),   except,     null,    null  ],
  // "#" : [count,     count,      take,       reshape,    take,       reshape,    null,    null  ],
  // "_" : [am(floor), am(floor),  drop,       ddrop,      drop,       cut,        null,    null  ],
  // "$" : [kfmt,      as(kfmt),   dfmt,       dfmt,       dfmt,       dfmt,       null,    null  ],
  // "?" : [real,      unique,     rnd,        pfind,      rnd,        ar(pfind),  splice,  null  ],
  // "@" : [type,      type,       atx,        atx,        atx,        atx,        amend4,  amend4],
  // "." : [keval,     keval,      call,       call,       call,       call,       dmend3,  dmend4],
  // "'" : [null,      null,       null,       bin,        null,       ar(bin),    null,    null  ],
  // "/" : [null,      null,       null,       null,       pack,       pack,       null,    null  ],
  // "\\": [null,      null,       null,       unpack,     split,      null,       null,    null  ],
  // "':": [null,      null,       null,       null,       kwindow,    null,       null,    null  ],
  let verbs: IndexMap<&str, (V1, V1, V2, V2, V2, V2, V3, V4)> = IndexMap::from([
    (
      ":",
      (
        v_ident as V1,
        v_ident as V1,
        v_d_colon as V2,
        v_d_colon as V2,
        v_d_colon as V2,
        v_d_colon as V2,
        v_none3 as V3,
        v_none4 as V4,
      ),
    ),
    ("+", (v_flip, v_flip, v_plus, v_plus, v_plus, v_plus, v_none3, v_none4)),
    ("-", (v_negate, v_negate, v_minus, v_minus, v_minus, v_minus, v_none3, v_none4)),
    ("*", (v_first, v_first, v_times, v_times, v_times, v_times, v_none3, v_none4)),
    ("%", (v_sqrt, v_sqrt, v_divide, v_divide, v_divide, v_divide, v_none3, v_none4)),
    ("!", (v_iota, v_odometer, v_d_bang, v_none2, v_d_bang, v_d_bang, v_none3, v_none4)),
  ]);

  match verbs.get(v) {
    Some((m_a, m_l, d_a_a, d_l_a, d_a_l, d_l_l, _triad, _tetrad)) => match (l, r) {
      (Some(KW::Noun(l)), KW::Noun(r)) => {
        if l.len() > 1 {
          if r.len() > 1 {
            d_l_l(l, r).and_then(|n| Ok(KW::Noun(n)))
          } else {
            d_l_a(l, r).and_then(|n| Ok(KW::Noun(n)))
          }
        } else {
          if r.len() > 1 {
            d_a_l(l, r).and_then(|n| Ok(KW::Noun(n)))
          } else {
            d_a_a(l, r).and_then(|n| Ok(KW::Noun(n)))
          }
        }
      }
      (None, KW::Noun(r)) => {
        if r.len() > 1 {
          m_l(r).and_then(|n| Ok(KW::Noun(n)))
        } else {
          m_a(r).and_then(|n| Ok(KW::Noun(n)))
        }
      }
      _ => {
        panic!("impossible")
      }
    },
    None => todo!("NotYetImplemented {}", v),
  }
}

// promote_nouns(l,r) => (l,r) eg. (Int, Bool) => (Int, Int)
// Similar to promote_num() can we combine these somehow and be more concise?
fn promote_nouns(l: K, r: K) -> (K, K) {
  match (&l, &r) {
    (K::Bool(l), K::Int(_)) => (K::Int(Some(*l as i64)), r),
    (K::Bool(l), K::Float(_)) => (K::Float(*l as f64), r),
    (K::Bool(l), K::IntArray(_)) => (K::IntArray(arr!([*l as i64])), r),
    (K::Bool(l), K::FloatArray(_)) => (K::FloatArray(arr!([*l as f64])), r),

    (K::Int(_), K::Bool(r)) => (l, K::Int(Some(*r as i64))),
    (K::Int(Some(l)), K::Float(_)) => (K::Float(*l as f64), r),
    (K::Int(Some(l)), K::BoolArray(r)) => {
      (K::IntArray(arr!([*l])), K::IntArray(r.cast(&DataType::Int64).unwrap()))
    }
    (K::Int(Some(l)), K::IntArray(_)) => (K::IntArray(arr!([*l])), r),
    (K::Int(Some(l)), K::FloatArray(_)) => (K::FloatArray(arr!([*l as f64])), r),
    (K::Int(None), K::Float(_)) => (K::Float(f64::NAN), r),
    (K::Int(None), K::BoolArray(_)) => (K::IntArray(arr!([None::<i64>])), r),
    (K::Int(None), K::IntArray(_)) => (K::IntArray(arr!([None::<i64>])), r),
    (K::Int(None), K::FloatArray(_)) => (K::FloatArray(arr!([f64::NAN])), r),

    (K::Float(_), K::Bool(r)) => (l, K::Float(*r as f64)),
    (K::Float(_), K::Int(Some(r))) => (l, K::Float(*r as f64)),
    (K::Float(_), K::Int(None)) => (l, K::Float(f64::NAN)),
    (K::Float(l), K::BoolArray(r)) => {
      (K::FloatArray(arr!([*l])), K::FloatArray(r.cast(&DataType::Float64).unwrap()))
    }
    (K::Float(l), K::IntArray(r)) => {
      (K::FloatArray(arr!([*l])), K::FloatArray(r.cast(&DataType::Float64).unwrap()))
    }
    (K::Float(l), K::FloatArray(_)) => (K::FloatArray(arr!([*l])), r),

    (K::BoolArray(_), K::Bool(r)) => (l, K::BoolArray(arr!([*r]))),
    (K::BoolArray(l), K::Int(Some(r))) => {
      (K::IntArray(l.cast(&DataType::Int64).unwrap()), K::IntArray(arr!([*r])))
    }
    (K::BoolArray(l), K::Int(None)) => {
      (K::IntArray(l.cast(&DataType::Int64).unwrap()), K::IntArray(arr!([None::<i64>])))
    }
    (K::BoolArray(l), K::Float(r)) => {
      (K::FloatArray(l.cast(&DataType::Float64).unwrap()), K::FloatArray(arr!([*r])))
    }
    (K::BoolArray(l), K::IntArray(_)) => (K::IntArray(l.cast(&DataType::Int64).unwrap()), r),
    (K::BoolArray(l), K::FloatArray(_)) => (K::FloatArray(l.cast(&DataType::Float64).unwrap()), r),

    (K::IntArray(_), K::Bool(r)) => (l, K::IntArray(arr!([*r as i64]))),
    (K::IntArray(_), K::Int(Some(r))) => (l, K::IntArray(arr!([*r as i64]))),
    (K::IntArray(_), K::Int(None)) => (l, K::IntArray(arr!([None::<i64>]))),
    (K::IntArray(_), K::Float(r)) => (l, K::FloatArray(arr!([*r as f64]))),
    (K::IntArray(_), K::BoolArray(r)) => (l, K::IntArray(r.cast(&DataType::Int64).unwrap())),
    (K::IntArray(l), K::FloatArray(_)) => (K::FloatArray(l.cast(&DataType::Float64).unwrap()), r),

    (K::FloatArray(_), K::Bool(r)) => (l, K::FloatArray(arr!([*r as f64]))),
    (K::FloatArray(_), K::Int(Some(r))) => (l, K::FloatArray(arr!([*r as f64]))),
    (K::FloatArray(_), K::Int(None)) => (l, K::FloatArray(arr!([f64::NAN]))),
    (K::FloatArray(_), K::Float(r)) => (l, K::FloatArray(arr!([*r as f64]))),
    (K::FloatArray(_), K::BoolArray(r)) => (l, K::FloatArray(r.cast(&DataType::Int64).unwrap())),
    (K::FloatArray(_), K::IntArray(r)) => (l, K::FloatArray(r.cast(&DataType::Float64).unwrap())),

    _ => (l, r),
  }
}

macro_rules! impl_op {
    ($op:tt, $opf:ident, $self:ident, $r:ident) => {
        match promote_nouns($self, $r) {
            (K::Bool(l), K::Bool(r)) => K::Int(Some(l as i64 $op r as i64)),
            (K::Int(Some(l)), K::Int(Some(r))) => K::Int(Some(l as i64 $op r)),
            (K::Int(None), K::Int(_)) | (K::Int(_), K::Int(None))=> K::Int(None),
            (K::Float(l), K::Float(r)) => K::Float(l $op r),
            (K::BoolArray(l), K::BoolArray(r)) => K::IntArray(l.cast(&DataType::Int64).unwrap() $op r.cast(&DataType::Int64).unwrap()),
            (K::IntArray(l), K::IntArray(r)) => K::IntArray(l $op r),
            (K::FloatArray(l), K::FloatArray(r)) => K::FloatArray(l $op r),
            (_, K::Dictionary(_)) => todo!("dict"),
            (K::Dictionary(_), _) => todo!("dict"),
            (_, K::Table(_)) => todo!("table"),
            (K::Table(_), _) => todo!("table"),
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
pub fn v_ident(x: K) -> Result<K, &'static str> { Ok(x) }
pub fn v_rident(_l: K, r: K) -> Result<K, &'static str> { Ok(r) }
pub fn v_flip(x: K) -> Result<K, &'static str> {
  match x {
    K::Dictionary(d) => {
      if d.iter().map(|(_k, v)| v.len()).all_equal() {
        let cols: Vec<Series> = d
          .iter()
          .map(|(k, v)| match v {
            K::SymbolArray(s)
            | K::BoolArray(s)
            | K::IntArray(s)
            | K::FloatArray(s)
            | K::CharArray(s) => Series::new(&k.to_string(), s.clone()),
            _ => todo!("handle atoms"),
          })
          .collect();
        Ok(K::Table(DataFrame::new(cols).unwrap()))
      } else {
        Err("length")
      }
    }
    _ => todo!("flip the rest"),
  }
}
macro_rules! atomicdyad {
  ($op:tt, $v:ident, $l:ident, $r:ident) => {
    match ($l, $r) {
      (K::Dictionary(ld), K::Dictionary(rd)) => {
        Ok(K::Dictionary(IndexMap::from_iter(ld.keys().chain(rd.keys()).unique().map(|k| {
          let a = ld.get(k);
          let b = rd.get(k);
          match (a, b) {
            (None, Some(b)) => (k.clone(), b.clone()),
            (Some(a), None) => (k.clone(), a.clone()),
            (Some(a), Some(b)) => (k.clone(), $v(a.clone(), b.clone()).unwrap()),
            (None, None) => panic!("impossible")
          }
        }))))
      }
      (K::Dictionary(ld), r) => {
        Ok(K::Dictionary(IndexMap::from_iter(ld.iter().map(|(k,v)| (k.clone(), $v(v.clone(), r.clone()).unwrap())))))
      }
      (l, K::Dictionary(rd)) => {
        Ok(K::Dictionary(IndexMap::from_iter(rd.iter().map(|(k,v)| (k.clone(), $v(l.clone(), v.clone()).unwrap())))))
      }
      (K::List(lv), K::List(rv)) => {
        Ok(K::List(zip(lv, rv).map(|(x, y)| $v(x.clone(), y.clone()).unwrap()).collect()))
      }
      (K::List(lv), r) => {
        Ok(K::List(lv.iter().map(|x| $v(x.clone(), r.clone()).unwrap()).collect()))
      }
      (l, K::List(rv)) => {
        Ok(K::List(rv.iter().map(|y| $v(l.clone(), y.clone()).unwrap()).collect()))
      }
      (l,r) => len_ok(&l, &r).and_then(|_| Ok(l $op r)),
    }
  };
}
pub fn v_plus(l: K, r: K) -> Result<K, &'static str> { atomicdyad!(+, v_plus, l, r) }
pub fn v_negate(x: K) -> Result<K, &'static str> { Ok(K::Int(Some(-1i64)) * x) }
pub fn v_minus(l: K, r: K) -> Result<K, &'static str> { atomicdyad!(-, v_minus, l, r) }
pub fn v_first(_x: K) -> Result<K, &'static str> { todo!("implement first") }
pub fn v_times(l: K, r: K) -> Result<K, &'static str> { atomicdyad!(*, v_times, l, r) }
pub fn v_sqrt(_x: K) -> Result<K, &'static str> { todo!("implement sqrt") }
pub fn v_divide(l: K, r: K) -> Result<K, &'static str> { atomicdyad!(/, v_divide,l, r) }
pub fn v_odometer(_r: K) -> Result<K, &'static str> { todo!("implement odometer") }
pub fn v_mod(_l: K, _r: K) -> Result<K, &'static str> { todo!("implement v_mod") }
pub fn v_iota(r: K) -> Result<K, &'static str> {
  debug!("v_iota");
  match r {
    K::Int(Some(i)) => Ok(K::IntArray(arr![(0..i).collect::<Vec<i64>>()])),
    _ => todo!("v_iota variants"),
  }
}

pub fn v_d_bang(l: K, r: K) -> Result<K, &'static str> {
  match l {
    K::SymbolArray(_) | K::Symbol(_) => v_makedict(l, r),
    _ => v_mod(l, r),
  }
}

fn strip_quotes(s: String) -> String {
  if s.starts_with("\"") && s.ends_with("\"") {
    s[1..s.len() - 1].into()
  } else {
    s
  }
}
pub fn v_makedict(l: K, r: K) -> Result<K, &'static str> {
  debug!("v_makedict() l: {:?}", l);
  debug!("v_makedict() r: {:?}", r);
  match l {
    K::SymbolArray(s) => match r {
      K::List(v) => {
        if s.len() == v.len() {
          Ok(K::Dictionary(IndexMap::from_iter(zip(
            s.iter().map(|s| strip_quotes(s.to_string())),
            v.iter().cloned(),
          ))))
        } else if v.len() == 1 {
          Ok(K::Dictionary(IndexMap::from_iter(zip(
            s.iter().map(|s| strip_quotes(s.to_string())),
            repeat(v[0].clone()),
          ))))
        } else {
          Err("length")
        }
      }
      K::BoolArray(_) | K::IntArray(_) | K::FloatArray(_) | K::CharArray(_) | K::SymbolArray(_) => {
        // `a`b`c!1 2 3 => `a`b`c!(1;2;3)
        Ok(K::Dictionary(IndexMap::from_iter(zip(
          s.iter().map(|s| strip_quotes(s.to_string())),
          k_to_vec(r).unwrap().iter().cloned(),
        ))))
      }
      _ => {
        if s.len() == 0 {
          Err("length")
        } else if s.len() == 1 {
          Ok(K::Dictionary(IndexMap::from([(strip_quotes(s.get(0).unwrap().to_string()), r)])))
        } else {
          Ok(K::Dictionary(IndexMap::from_iter(zip(s.iter().map(|s| strip_quotes(s.to_string())), repeat(r)))))
        }
      }
    },
    K::Symbol(s) => match r {
      _ => Ok(K::Dictionary(IndexMap::from([(s, r)]))),
    },
    _ => {
      todo!("modulo")
    }
  }
}

pub fn v_colon(_r: K) -> Result<K, &'static str> { todo!(": monad") }
pub fn v_d_colon(l: K, r: K) -> Result<K, &'static str> {
  println!("l: {:?}, r: {:?}", l, r);
  match (&l, &r) {
    (K::Int(Some(2i64)), K::Symbol(s)) => {
      let p = Path::new(&s);
      if p.exists() {
        match Path::new(&s).extension() {
          Some(e) => {
            if e == "csv" {
              Ok(K::Table(CsvReader::from_path(p).unwrap().has_header(true).finish().unwrap()))
            } else if e == "parquet" {
              // let lf1 = LazyFrame::scan_parquet(p, Default::default()).unwrap();
              Ok(K::Table(ParquetReader::new(File::open(p).unwrap()).finish().unwrap()))
            } else {
              todo!("other file types")
            }
          }
          _ => todo!("no extension"),
        }
      } else {
        Err("path does not exist")
      }
    }
    _ => v_rident(l, r),
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
          let nouns: VecDeque<KW> = stack
            .drain(..i + 1)
            .filter(|w| !matches!(*w, KW::SC) && !matches!(*w, KW::RP))
            .collect();
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
    Err("parse error")
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
      'a'..='z' | 'A'..='Z' => {
        let (j, k) = scan_name(&code[i..]).unwrap();
        words.push(k);
        skip = j;
    }
      _ => return Err("TODO: scan()"),
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
  // read until first char outside [a-z0-9._`]
  // split on ` or space`
  //
  // a SymbolArray *potentially* extends until the first symbol character
  let sentence = match code.find(|c: char| {
    !(c.is_ascii_alphanumeric() || c.is_ascii_whitespace() || ['.', '_', '`'].contains(&c))
  }) {
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
      1 => return Ok((i - 1, KW::Noun(K::Symbol(ss[0].clone())))),
      _ => {
        return Ok((
          i - 1,
          KW::Noun(K::SymbolArray(
            Series::new("a", ss).cast(&DataType::Categorical(None)).unwrap(),
          )),
        ))
      }
    }
  }
}
pub fn scan_name(code: &str) -> Result<(usize, KW), &'static str> {
  // read a single Name
  // a Name extends until the first symbol character or space
  let sentence = match code.find(|c: char| {
    !(c.is_ascii_alphanumeric() || ['.'].contains(&c))
  }) {
    Some(c) => &code[..c],
    None => code,
  };
  return Ok((sentence.len(), KW::Name(sentence.into())))
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
