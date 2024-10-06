use crate::*;
use rand::distributions::{Distribution, Uniform};

pub fn v_imat(_x: K) -> Result<K, &'static str> { Err("nyi") }
pub fn v_group(_x: K) -> Result<K, &'static str> { Err("nyi") }
pub fn v_equal(x: K, y: K) -> Result<K, &'static str> {
  if x.len() != y.len() {
    debug!("x.len(): {}, y.len(): {}", x.len(), y.len());
    Err("length")
  } else {
    match promote_nouns(x, y) {
      (K::Bool(l), K::Bool(r)) => Ok(K::Bool((l == r) as u8)),
      (K::Int(Some(l)), K::Int(Some(r))) => Ok(K::Bool((l == r) as u8)),
      (K::Int(None), K::Int(_)) | (K::Int(_), K::Int(None)) => Ok(K::Bool(0)),
      (K::Float(l), K::Float(r)) => Ok(K::Bool((l == r) as u8)),
      (K::BoolArray(l), K::BoolArray(r)) => Ok(K::BoolArray(l.equal(&r).unwrap().into())),
      (K::IntArray(l), K::IntArray(r)) => Ok(K::BoolArray(l.equal(&r).unwrap().into())),
      (K::FloatArray(l), K::FloatArray(r)) => Ok(K::BoolArray(l.equal(&r).unwrap().into())),
      (K::List(l), K::List(r)) => Ok(K::BoolArray(arr!(zip(l.iter(), r.iter())
        .map(|(l, r)| {
          let (l, r) = promote_nouns(l.clone(), r.clone());
          l == r
        })
        .collect::<Vec<bool>>()))),
      (K::Dictionary(l), K::Dictionary(r)) => {
        Ok(K::Dictionary(IndexMap::from_iter(l.keys().filter_map(|k| {
          if r.keys().contains(k) {
            let (l, r) = promote_nouns(l.get(k).unwrap().clone(), r.get(k).unwrap().clone());
            Some((k.clone(), K::Bool((l == r) as u8)))
          } else {
            None
          }
        }))))
      }
      (_, K::Table(_)) => todo!("table"),
      (K::Table(_), _) => todo!("table"),
      _ => Err("nyi"),
    }
  }
}

pub fn v_count(x: K) -> Result<K, &'static str> { Ok(K::Int(Some(x.len().try_into().unwrap()))) }
pub fn v_take(_l: K, _r: K) -> Result<K, &'static str> { Err("nyi") }

#[macro_export]
macro_rules! reshape_atom_by_type {
  ($a:ident, $type:path, $rev_shape:ident) => {{
    let mut result: K = $type(arr!([$a].repeat($rev_shape[0] as usize)));
    for i in $rev_shape.iter().skip(1) {
      result = K::List((0..*i).map(|_| result.clone()).collect_vec());
    }
    Ok(result)
  }};
}

pub fn v_reshape(l: K, r: K) -> Result<K, &'static str> {
  match l {
    K::IntArray(a) => {
      let rev_shape: Vec<i64> =
        a.i64().unwrap().to_vec().iter().rev().map(|i| i.unwrap()).collect();
      match r {
        K::Bool(b) => {
          reshape_atom_by_type!(b, K::BoolArray, rev_shape)
        }
        K::Int(None) => Err("nyi"),
        K::Int(Some(i)) => {
          reshape_atom_by_type!(i, K::IntArray, rev_shape)
        }
        K::Float(f) => {
          reshape_atom_by_type!(f, K::FloatArray, rev_shape)
        }
        K::Char(_) => Err("nyi"),
        K::Symbol(_) => Err("nyi"),

        K::SymbolArray(_)
        | K::BoolArray(_)
        | K::IntArray(_)
        | K::FloatArray(_)
        | K::CharArray(_)
        | K::List(_)
        | K::Dictionary(_)
        | K::Table(_)
        | K::Nil => Err("nyi"),
        K::Name(_) => panic!("impossible"),
      }
    }
    K::BoolArray(_) => Err("nyi"),
    _ => Err("type"),
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
            K::SymbolArray(s) | K::BoolArray(s) | K::IntArray(s) | K::FloatArray(s) => {
              Series::new(&k.to_string(), s.clone())
            }
            // | K::CharArray(s) => Series::new(&k.to_string(), s.clone()),
            K::CharArray(s) => Series::new(&k.to_string(), s),
            K::List(v) => {
              if v.iter().all(|i| match i {
                K::CharArray(_) => true,
                // K::Char(_) => true, // TODO
                _ => false,
              }) {
                let vs: Vec<String> = v.iter().map(|s| s.to_string()).collect();
                Series::new(&k.to_string(), vs.clone())
              } else {
                // Err("type")
                panic!("type error?")
              }
            }
            _ => todo!("handle atoms"),
          })
          .collect();
        Ok(K::Table(DataFrame::new(cols).unwrap()))
      } else {
        Err("length")
      }
    }
    K::Table(df) => {
      let k = Series::new("a", df.get_column_names())
        .cast(&DataType::Categorical(None, CategoricalOrdering::Lexical))
        .unwrap();
      let v: Vec<K> = df
        .get_columns()
        .iter()
        .map(|s| match s.dtype() {
          DataType::Boolean => K::BoolArray(s.clone()),
          DataType::Int64 => K::IntArray(s.clone()),
          DataType::Float64 => K::FloatArray(s.clone()),
          // DataType::String => K::CharArray(s.clone()), // TODO K::List([K::CharArray(), ...])
          DataType::Categorical { .. } => K::SymbolArray(s.clone()),
          _ => panic!("impossible"),
        })
        .collect();

      Ok(K::Dictionary(IndexMap::from_iter(zip(
        k.iter().map(|s| strip_quotes(s.to_string())),
        v.iter().cloned(),
      ))))
    }
    _ => todo!("flip the rest"),
  }
}
macro_rules! atomicdyad {
  ($op:tt, $v:ident, $named_op:ident, $l:ident, $r:ident) => {
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
      (K::Table(_lt), K::Table(_rt)) => todo!("table"),
      (K::Table(df), K::Bool(i)) => {
        Ok(K::Table(DataFrame::new(df.iter().map(|s| s $op i ).collect::<Vec<Series>>()).unwrap()))
      }
      (K::Table(df), K::Int(i)) => {
        Ok(K::Table(DataFrame::new(df.iter().map(|s| s $op i.unwrap()).collect::<Vec<Series>>()).unwrap()))
      }
      (K::Table(df), K::Float(f)) => {
        Ok(K::Table(DataFrame::new(df.iter().map(|s| s.to_float().unwrap() $op f).collect::<Vec<Series>>()).unwrap()))
      }
      (K::Table(_df), _r) => todo!("nyi"),
      (K::Bool(i), K::Table(df)) => {
        Ok(K::Table(DataFrame::new(df.iter().map(|s| i.$named_op(&s)).collect::<Vec<Series>>()).unwrap()))
      }
      (K::Int(i), K::Table(df)) => {
        Ok(K::Table(DataFrame::new(df.iter().map(|s| i.unwrap().$named_op(&s)).collect::<Vec<Series>>()).unwrap()))
      }
      (K::Float(f), K::Table(df)) => {
        Ok(K::Table(DataFrame::new(df.iter().map(|s| f.$named_op(&s.to_float().unwrap())).collect::<Vec<Series>>()).unwrap()))
      }
      (_l, K::Table(_df)) => {
        todo!("nyi")
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
pub fn v_plus(l: K, r: K) -> Result<K, &'static str> { atomicdyad!(+, v_plus, add, l, r) }
pub fn v_negate(x: K) -> Result<K, &'static str> { Ok(K::Int(Some(-1i64)) * x) }
pub fn v_minus(l: K, r: K) -> Result<K, &'static str> { atomicdyad!(-, v_minus, sub, l, r) }
pub fn v_first(_x: K) -> Result<K, &'static str> { todo!("implement first") }
pub fn v_times(l: K, r: K) -> Result<K, &'static str> {
  match (l.clone(), r.clone()) {
    // TODO can we make this less repetitive and explicit?
    (K::Int(i), K::Table(df)) => Ok(K::Table(
      DataFrame::new(df.iter().map(|s| i.unwrap().mul(s)).collect::<Vec<Series>>()).unwrap(),
    )),
    _ => atomicdyad!(*, v_times, mul, l, r),
  }
}
pub fn v_sqrt(_x: K) -> Result<K, &'static str> { todo!("implement sqrt") }
pub fn v_divide(l: K, r: K) -> Result<K, &'static str> { atomicdyad!(/, v_divide, div, l, r) }
pub fn v_odometer(_r: K) -> Result<K, &'static str> { todo!("implement odometer") }
pub fn v_mod(l: K, r: K) -> Result<K, &'static str> {
  match (l, r) {
    (K::Int(Some(i)), K::IntArray(a)) => Ok(K::IntArray(a % i)),
    (K::Int(Some(i)), K::FloatArray(a)) => Ok(K::FloatArray(a % i)),
    (K::Int(Some(_i)), K::CharArray(_a)) => todo!("implement v_mod"),
    _ => todo!("implement v_mod"),
  }
}

pub fn v_where(_r: K) -> Result<K, &'static str> { Err("nyi") }
pub fn v_min(_l: K, _r: K) -> Result<K, &'static str> { Err("nyi") }

pub fn v_reverse(_r: K) -> Result<K, &'static str> { Err("nyi") }
pub fn v_max(_l: K, _r: K) -> Result<K, &'static str> { Err("nyi") }

pub fn v_asc(_r: K) -> Result<K, &'static str> { Err("nyi") }
pub fn v_lesser(_l: K, _r: K) -> Result<K, &'static str> { Err("nyi") }

pub fn v_desc(_r: K) -> Result<K, &'static str> { Err("nyi") }
pub fn v_greater(_l: K, _r: K) -> Result<K, &'static str> { Err("nyi") }

pub fn v_not(_r: K) -> Result<K, &'static str> { Err("nyi") }
pub fn v_match(_l: K, _r: K) -> Result<K, &'static str> { Err("nyi") }

pub fn v_enlist(_r: K) -> Result<K, &'static str> { Err("nyi") }
pub fn v_concat(_l: K, _r: K) -> Result<K, &'static str> { Err("nyi") }

pub fn v_isnull(_r: K) -> Result<K, &'static str> { Err("nyi") }
pub fn v_fill(_l: K, _r: K) -> Result<K, &'static str> { Err("nyi") }
pub fn v_except(_l: K, _r: K) -> Result<K, &'static str> { Err("nyi") }

pub fn v_floor(_r: K) -> Result<K, &'static str> { Err("nyi") }
pub fn v_drop(_l: K, _r: K) -> Result<K, &'static str> { Err("nyi") }
pub fn v_delete(_l: K, _r: K) -> Result<K, &'static str> { Err("nyi") }
pub fn v_cut(_l: K, _r: K) -> Result<K, &'static str> { Err("nyi") }

pub fn v_string(_r: K) -> Result<K, &'static str> { Err("nyi") }
pub fn v_dfmt(_l: K, _r: K) -> Result<K, &'static str> { Err("nyi") }
pub fn v_pad(_l: K, _r: K) -> Result<K, &'static str> { Err("nyi") }
pub fn v_cast(_l: K, _r: K) -> Result<K, &'static str> { Err("nyi") }

pub fn v_randfloat(r: K) -> Result<K, &'static str> {
  match r {
    K::Int(Some(0)) => Err("nyi"),
    K::Int(Some(i)) if i < 0 => Err("domain"),
    K::Int(Some(i)) if i > 0 => Ok(K::FloatArray(
      ChunkedArray::<Float64Type>::rand_uniform("", i as usize, 0.0f64, 1.0f64).into_series(),
    )),
    _ => Err("type"),
  }
}
pub fn v_unique(r: K) -> Result<K, &'static str> {
  debug!("v_unique({:?})", r);

  match r {
    K::SymbolArray(a) => Ok(K::SymbolArray(a.unique().unwrap())),
    K::BoolArray(a) => Ok(K::BoolArray(a.unique().unwrap())),
    K::IntArray(a) => Ok(K::IntArray(a.unique().unwrap())),
    K::FloatArray(a) => Ok(K::FloatArray(a.unique().unwrap())),
    K::CharArray(a) => Ok(K::CharArray(a.chars().unique().collect())),
    // TODO ?(3.14;"abc";3.14) works in ngn/k but k9 throws domain error if the list has any float item and otherwise works.
    // K::List(v) => Ok(K::List(v.into_iter().unique().collect())),
    K::List(_v) => Err("nyi: v_unique(K::List(_))"),
    _ => Err("domain"), //
  }
}
pub fn v_rand(l: K, r: K) -> Result<K, &'static str> {
  match (l.clone(), r.clone()) {
    (K::Int(Some(x)), K::Int(Some(y))) if x > 0 => {
      let mut rng = rand::thread_rng();
      let range = Uniform::from(0..y);

      let v: Vec<i64> = (0..x).map(|_i| range.sample(&mut rng)).collect();
      Ok(K::IntArray(Series::new("", v)))
    }
    (K::Int(Some(x)), K::Int(Some(_y))) if x < 0 => {
      todo!("nyi v_rand with no repeats")
    }
    (K::Int(Some(_x)), y) if y.len() > 1 => {
      let idxs = v_rand(l, K::Int(Some(y.len() as i64))).unwrap();
      v_at(r, idxs)
    }
    _ => Err("nyi"),
  }
}
pub fn v_find(_l: K, _r: K) -> Result<K, &'static str> { Err("nyi") }
pub fn v_splice(_x: K, _y: K, _z: K) -> Result<K, &'static str> { Err("nyi") }

pub fn v_type(r: K) -> Result<K, &'static str> {
  use K::*;
  // TODO allow checking type of KW::Verb etc, see ngn/k's types help \0
  match r {
    Bool(_) => Ok(Symbol("i".to_string())),
    Int(_) => Ok(Symbol("i".to_string())),
    Float(_) => Ok(Symbol("f".to_string())),
    Char(_) => Ok(Symbol("c".to_string())),
    Symbol(_) => Ok(Symbol("s".to_string())),
    SymbolArray(_) => Ok(Symbol("S".to_string())),
    BoolArray(_) => Ok(Symbol("I".to_string())),
    IntArray(_) => Ok(Symbol("I".to_string())),
    FloatArray(_) => Ok(Symbol("F".to_string())),
    CharArray(_) => Ok(Symbol("C".to_string())),
    Nil => Ok(Symbol("A".to_string())),
    List(_) => Ok(Symbol("A".to_string())),
    Dictionary(_) => Ok(Symbol("m".to_string())),
    Table(_) => Ok(Symbol("M".to_string())),
    Name(_) => panic!("impossible"),
  }
}

pub fn v_at(l: K, r: K) -> Result<K, &'static str> {
  match r {
    K::Int(None) => match l.clone() {
      K::SymbolArray(_) | K::BoolArray(_) | K::IntArray(_) | K::FloatArray(_) | K::CharArray(_) => {
        l.fill(0)
      }
      K::List(_v) => todo!("v_at"),
      K::Dictionary(_d) => todo!("v_at"),
      K::Table(_df) => todo!("v_at"),
      _ => Err("type"),
    },
    K::Bool(i) => {
      // TODO remove this code duplication of K::Int(_) case below
      if (i as usize) < l.len() {
        match l.clone() {
          K::IntArray(a) => Ok(K::Int(Some(a.i64().unwrap().get(i as usize).unwrap()))),
          K::FloatArray(a) => Ok(K::Float(a.f64().unwrap().get(i as usize).unwrap())),
          K::CharArray(a) => {
            let c: char = a.chars().nth(i.into()).unwrap();
            Ok(K::Char(c))
          }
          K::List(v) => {
            if (i as usize) < v.len() {
              Ok(v[i as usize].clone())
            } else {
              Err("length")
            }
          }
          _ => todo!("index into l"),
        }
      } else {
        l.fill(0)
      }
    }
    K::Int(Some(i)) => {
      if i >= 0 && (i as usize) < l.len() {
        match l.clone() {
          K::IntArray(a) => Ok(K::Int(Some(a.i64().unwrap().get(i as usize).unwrap()))),
          K::FloatArray(a) => Ok(K::Float(a.f64().unwrap().get(i as usize).unwrap())),
          K::CharArray(a) => {
            let c: char = a.chars().nth(i as usize).unwrap();
            Ok(K::Char(c))
          }
          K::List(_v) => todo!("v_at List"),
          _ => todo!("index into l"),
        }
      } else {
        l.fill(0)
      }
    }
    K::BoolArray(i) => v_at(l, K::try_from(i.cast(&DataType::Int64).unwrap()).unwrap()),
    K::IntArray(i) => {
      // https://docs.rs/polars/latest/polars/series/struct.Series.html#method.take_threaded
      // Notes: Out of bounds access doesn’t Error but will return a Null value
      // TODO Add fills not Nulls
      let i = Series::new(
        "",
        i.i64()
          .unwrap()
          .into_iter()
          .map(|i| i.unwrap_or(4_294_967_295) as u32)
          .collect::<Vec<u32>>(),
      );
      let idcs: Vec<u32> = i.u32().unwrap().into_iter().map(|i| i.unwrap()).collect::<Vec<u32>>();
      match l.clone() {
        K::SymbolArray(a) => match a.take_slice(&idcs) {
          Ok(a) => Ok(K::SymbolArray(a)),
          _ => todo!("index out of bounds - this shouldn't be an error"),
        },
        K::BoolArray(a) => match a.take_slice(&idcs) {
          Ok(a) => Ok(K::BoolArray(a)),
          _ => todo!("index out of bounds - this shouldn't be an error"),
        },
        K::IntArray(a) => match a.take_slice(&idcs) {
          Ok(a) => Ok(K::IntArray(a)),
          _ => todo!("index out of bounds - this shouldn't be an error"),
        },
        K::FloatArray(a) => match a.take_slice(&idcs) {
          Ok(a) => Ok(K::FloatArray(a)),
          _ => todo!("index out of bounds - this shouldn't be an error"),
        },
        K::CharArray(a) => Ok(K::CharArray(
          i.u32()
            .unwrap()
            .iter()
            .map(|i| a.chars().nth(i.unwrap() as usize).unwrap_or(' '))
            .collect(),
        )),
        K::List(_) => todo!("v_at K::List"),
        _ => todo!("v_at"),
      }
    }
    K::Symbol(s) => match l.clone() {
      K::Dictionary(d) => {
        if d.contains_key(&s) {
          Ok(d.get(&s).unwrap().clone())
        } else {
          Ok(K::Nil) // TODO Is this the same behaviour as ngn/k and k9?
        }
      }
      K::Table(df) => match df.get_column_index(&s) {
        Some(i) => K::try_from(df[i].clone()),
        _ => todo!("nyi"),
      },
      K::SymbolArray(ss) => match l.clone() {
        K::Dictionary(d) => {
          Ok(K::List(
            ss.categorical()
              .unwrap()
              .iter_str()
              .map(|s| {
                if d.contains_key(s.unwrap()) {
                  d.get(s.unwrap()).unwrap().clone()
                } else {
                  K::Nil // TODO Is this the same behaviour as ngn/k and k9?
                }
              })
              .collect::<Vec<K>>(),
          ))
        }
        K::Table(_df) => todo!("nyi"),
        _ => todo!("nyi"),
      },
      _ => Err("type"),
    },
    K::SymbolArray(s) => {
      let keys: Vec<K> = s
        .iter()
        .map(|s| {
          let s = s.to_string();
          K::Symbol(s[1..s.len() - 1].to_string())
        })
        .collect();
      Ok(K::List(keys.into_iter().map(|k| v_at(l.clone(), k).unwrap()).collect()))
    }
    _ => todo!("v_at({:?}, {:?})", l, r),
  }
}

// https://k.miraheze.org/wiki/Amend
pub fn v_amend3(_x: K, _y: K, _z: K) -> Result<K, &'static str> { Err("nyi") }
pub fn v_amend4(_x: K, _y: K, _f: K, _z: K) -> Result<K, &'static str> { Err("nyi") }

pub fn v_eval(_r: K) -> Result<K, &'static str> { Err("nyi") }
pub fn v_dot(_l: K, _r: K) -> Result<K, &'static str> { Err("nyi") }
// https://k.miraheze.org/wiki/Deep_amend
pub fn v_deepamend3(_x: K, _y: K, _z: K) -> Result<K, &'static str> { Err("nyi") }
pub fn v_deepamend4(_x: K, _y: K, _f: K, _z: K) -> Result<K, &'static str> { Err("nyi") }
pub fn v_try(_x: K, _y: K, _z: K) -> Result<K, &'static str> { Err("nyi") }

pub fn v_join(l: K, r: K) -> Result<K, &'static str> {
  match l {
    K::Char(l) => match r {
      K::List(v) => {
        let v: Option<Vec<String>> =
          v.iter().map(|k| if let K::CharArray(s) = k { Some(s.clone()) } else { None }).collect();
        match v {
          Some(v) => Ok(K::CharArray(v.join(&l.to_string()))),
          None => Err("type"),
        }
      }
      _ => Err("type"),
    },
    _ => Err("type"),
  }
}
pub fn v_unpack(_l: K, _r: K) -> Result<K, &'static str> { Err("nyi") }
pub fn v_split(l: K, r: K) -> Result<K, &'static str> {
  match l {
    K::Char(l) => match r {
      K::CharArray(r) => Ok(K::List(r.split(l).map(|s| K::CharArray(s.to_string())).collect())),
      _ => Err("type"),
    },
    K::CharArray(l) => match r {
      K::CharArray(r) => Ok(K::List(r.split(&l).map(|s| K::CharArray(s.to_string())).collect())),
      _ => Err("type"),
    },
    _ => Err("type"),
  }
}

pub fn v_iota(r: K) -> Result<K, &'static str> {
  debug!("v_iota");
  match r {
    K::Int(Some(i)) => Ok(K::IntArray(arr![(0..i).collect::<Vec<i64>>()])),
    _ => todo!("v_iota variants"),
  }
}
pub fn v_sum(x: K) -> Result<K, &'static str> {
  match x {
    K::BoolArray(a) => Ok(K::Int(a.sum().ok())),
    K::IntArray(a) => Ok(K::Int(a.sum().ok())),
    K::FloatArray(a) => Ok(K::Float(a.sum().unwrap_or(f64::NAN))),
    _ => {
      // Fall back to slow catchall. TODO Something nicer
      let mut env = Env { names: HashMap::new(), parent: None };
      let sum = eval(&mut env, scan("{x+y}/").unwrap()).unwrap();
      Ok(eval(&mut env, vec![sum, KW::Noun(x.clone())]).unwrap().unwrap_noun())
    }
  }
}
pub fn v_d_sum(l: K, r: K) -> Result<K, &'static str> { Ok(l + v_sum(r).unwrap()) }

// TODO
// pub fn v_product(x: K) -> Result<K, &'static str> {
//   match x {
//     K::BoolArray(a) => Ok(K::Int(a.product().ok().into())),
//     K::IntArray(a) => Ok(K::Int(a.product().ok())),
//     K::FloatArray(a) => Ok(K::Float(a.product().unwrap_or(f64::NAN))),
//     _ => todo!("other types of K"),
//   }
// }
// pub fn v_d_product(l: K, r: K) -> Result<K, &'static str> { Ok(l + v_product(r).unwrap()) }

pub fn v_d_bang(l: K, r: K) -> Result<K, &'static str> {
  match l {
    K::SymbolArray(_) | K::Symbol(_) => v_makedict(l, r),
    _ => v_mod(l, r),
  }
}

pub fn v_each(env: &mut Env, v: KW, x: K) -> Result<K, &'static str> {
  match v {
    f @ KW::Verb { .. } | f @ KW::Function { .. } => k_to_vec(x).map(|v| {
      let r: Vec<K> = v
        .iter()
        .cloned()
        .map(|y|
             // apply_primitive(env, &name, None, KW::Noun(y.clone())).unwrap().unwrap_noun()
             eval(env, vec![f.clone(), KW::Noun(y.clone())]).unwrap().unwrap_noun())
        .collect();
      promote_num(r.clone()).unwrap_or(K::List(r))
    }),
    _ => Err("type"),
  }
}
pub fn v_d_each(_env: &mut Env, _v: KW, _x: K, _y: K) -> Result<K, &'static str> { todo!("each") }

// TODO dispatch / based on inputs:  over | fixedpoint
pub fn a_slash(env: &mut Env, v: KW, x: K) -> Result<K, &'static str> {
  match v.clone() {
    KW::Verb { .. } => todo!("figure out if v:{} is monadic or dyadic", v),
    KW::Function { body, args, adverb } => match args.len() {
      2 => v_fold(env, v, x),
      1 => todo!("v_fixedpoint(env, v, x)"),
      _ => Err("rank"),
    },
    _ => panic!("impossible"),
  }
}

pub fn a_d_slash(_env: &mut Env, _v: KW, _x: K, _y: K) -> Result<K, &'static str> {
  // TODO check the rank of v and type of x and handle the different meanings of /
  // https://k.miraheze.org/wiki/For
  // https://k.miraheze.org/wiki/While
  // https://k.miraheze.org/wiki/Base_encode
  todo!("for/while/join/base_encode")
}

pub fn v_fold(env: &mut Env, v: KW, x: K) -> Result<K, &'static str> {
  // split into list, then reduce
  // TODO check the rank of v and handle the different meanings of /
  // https://k.miraheze.org/wiki/Fold
  // https://k.miraheze.org/wiki/Fixedpoint
  //
  match v {
    f @ KW::Verb { .. } | f @ KW::Function { .. } => k_to_vec(x).and_then(|v| {
      let r = v.iter().cloned().reduce(|x, y| {
        // apply_primitive(env, &name, Some(KW::Noun(x.clone())), KW::Noun(y.clone())).unwrap().unwrap_noun()
        eval(
          env,
          vec![f.clone(), KW::FuncArgs(vec![vec![KW::Noun(x.clone())], vec![KW::Noun(y.clone())]])],
        )
        .unwrap()
        .unwrap_noun()
      });
      match r {
        Some(k) => Ok(k.clone()),
        None => Err("TODO not sure what this error case is"),
      }
    }),
    _ => Err("type"),
  }
}
pub fn v_d_fold(env: &mut Env, v: KW, x: K, y: K) -> Result<K, &'static str> {
  // TODO check the rank of v and type of x and handle the different meanings of /
  // https://k.miraheze.org/wiki/For
  // https://k.miraheze.org/wiki/While
  // https://k.miraheze.org/wiki/Join
  // https://k.miraheze.org/wiki/Base_encode
  if let KW::Verb { ref name } = v {
    let mut e = Env { names: HashMap::new(), parent: Some(Box::new(env.clone())) }; // TODO This will lose names if the fold verb does global assignment
    Ok(
      apply_primitive(
        env,
        &name.clone(),
        Some(KW::Noun(x.clone())),
        KW::Noun(v_fold(&mut e, v, y).unwrap()),
      )
      .unwrap()
      .unwrap_noun(),
    )
  } else {
    Err("type")
  }
}
pub fn v_scan(_env: &mut Env, _v: KW, _x: K) -> Result<K, &'static str> { todo!("scan") }
pub fn v_d_scan(_env: &mut Env, _v: KW, _x: K, _y: K) -> Result<K, &'static str> { todo!("scan") }

pub fn v_eachprior(_env: &mut Env, _v: KW, _x: K) -> Result<K, &'static str> { todo!("scan") }
pub fn v_windows(_env: &mut Env, _v: KW, _x: K, _y: K) -> Result<K, &'static str> { todo!("scan") }
pub fn v_eachright(_env: &mut Env, _v: KW, _x: K) -> Result<K, &'static str> { todo!("scan") }
pub fn v_eachleft(_env: &mut Env, _v: KW, _x: K) -> Result<K, &'static str> { todo!("scan") }

pub fn strip_quotes(s: String) -> String {
  if s.starts_with('\"') && s.ends_with('\"') {
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
          Ok(K::Dictionary(IndexMap::from_iter(zip(
            s.iter().map(|s| strip_quotes(s.to_string())),
            repeat(r),
          ))))
        }
      }
    },
    K::Symbol(s) => Ok(K::Dictionary(IndexMap::from([(s, r)]))),
    _ => {
      todo!("modulo")
    }
  }
}

pub fn v_colon(_r: K) -> Result<K, &'static str> { todo!(": monad") }
pub fn v_d_colon(env: &mut Env, l: K, r: KW) -> Result<KW, &'static str> {
  debug!("l: {:?}, r: {:?}", l, r);
  match (&l, &r) {
    (K::Bool(0), KW::Noun(K::CharArray(a))) => Ok(KW::Noun(K::List(
      std::fs::read_to_string(a).unwrap().lines().map(String::from).map(K::from).collect(),
    ))),
    (K::Int(Some(2i64)), KW::Noun(K::Symbol(s))) => {
      let p = Path::new(&s);
      if p.exists() {
        match Path::new(&s).extension() {
          Some(e) => {
            if e == "csv" {
              Ok(KW::Noun(K::Table(
                CsvReadOptions::default()
                  .with_has_header(true)
                  .try_into_reader_with_file_path(Some(p.to_path_buf()))
                  .unwrap()
                  .finish()
                  .unwrap(),
              )))
            } else if e == "parquet" {
              // let lf1 = LazyFrame::scan_parquet(p, Default::default()).unwrap();
              Ok(KW::Noun(K::Table(ParquetReader::new(File::open(p).unwrap()).finish().unwrap())))
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
    (K::Name(n), r) => {
      env.names.extend([(n.clone(), r.clone())]);
      Ok(r.clone())
    }
    (_, KW::Noun(r)) => Ok(KW::Noun(v_rident(l, r.clone()).unwrap())),
    _ => panic!("impossible"),
  }
}

pub fn v_prm(_r: K) -> Result<K, &'static str> { Err("nyi") }
pub fn v_in(_l: K, _r: K) -> Result<K, &'static str> { Err("nyi") }
pub fn v_has(_l: K, _r: K) -> Result<K, &'static str> { Err("nyi") }
pub fn v_within(_l: K, _r: K) -> Result<K, &'static str> { Err("nyi") }
