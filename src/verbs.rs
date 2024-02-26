use crate::*;
use itertools::Itertools;

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
pub fn v_reshape(_l: K, _r: K) -> Result<K, &'static str> { Err("nyi") }

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

pub fn v_randfloat(_r: K) -> Result<K, &'static str> { Err("nyi") }
pub fn v_unique(r: K) -> Result<K, &'static str> {
  debug!("v_unique({:?})", r);

  match r {
    K::SymbolArray(a) => Ok(K::SymbolArray(a.unique().unwrap())),
    K::BoolArray(a) => Ok(K::BoolArray(a.unique().unwrap())),
    K::IntArray(a) => Ok(K::IntArray(a.unique().unwrap())),
    K::FloatArray(a) => Ok(K::FloatArray(a.unique().unwrap())),
    K::CharArray(a) => Ok(K::CharArray(a.unique().unwrap())),
    // TODO ?(3.14;"abc";3.14) works in ngn/k but k9 throws domain error if the list has any float item and otherwise works.
    // K::List(v) => Ok(K::List(v.into_iter().unique().collect())),
    K::List(_v) => Err("nyi: v_unique(K::List(_))"),
    _ => Err("domain"), //
  }
}
pub fn v_rand(_l: K, _r: K) -> Result<K, &'static str> { Err("nyi") }
pub fn v_find(_l: K, _r: K) -> Result<K, &'static str> { Err("nyi") }
pub fn v_splice(_x: K, _y: K, _z: K) -> Result<K, &'static str> { Err("nyi") }

pub fn v_type(_r: K) -> Result<K, &'static str> { Err("nyi") }
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
            // TODO std::str::from_utf8()...
            // let s = a.utf8().unwrap().get(i as usize).unwrap();
            let s = std::str::from_utf8(
              &a.cast(&DataType::UInt8)
                .unwrap()
                .u8()
                .unwrap()
                .into_iter()
                .map(|u| match u {
                  Some(u) => u,
                  None => panic!("impossible"),
                })
                .collect::<Vec<u8>>(),
            )
            .unwrap()
            .to_string();
            let c: char = s.chars().nth(i.into()).unwrap();
            Ok(K::Char(c))
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
            // TODO std::str::from_utf8()...
            // let s = a.utf8().unwrap().get(i as usize).unwrap();
            let s = std::str::from_utf8(
              &a.cast(&DataType::UInt8)
                .unwrap()
                .u8()
                .unwrap()
                .into_iter()
                .map(|u| match u {
                  Some(u) => u,
                  None => panic!("impossible"),
                })
                .collect::<Vec<u8>>(),
            )
            .unwrap()
            .to_string();
            let c: char = s.chars().nth(i as usize).unwrap();
            Ok(K::Char(c))
          }
          _ => todo!("index into l"),
        }
      } else {
        l.fill(0)
      }
    }
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
      match l.clone() {
        K::SymbolArray(a) => Ok(K::SymbolArray(a.take_threaded(i.u32().unwrap(), true).unwrap())),
        K::BoolArray(a) => Ok(K::BoolArray(a.take_threaded(i.u32().unwrap(), true).unwrap())),
        K::IntArray(a) => Ok(K::IntArray(a.take_threaded(i.u32().unwrap(), true).unwrap())),
        K::FloatArray(a) => Ok(K::FloatArray(a.take_threaded(i.u32().unwrap(), true).unwrap())),
        K::CharArray(a) => Ok(K::CharArray(a.take_threaded(i.u32().unwrap(), true).unwrap())),
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
      K::Table(_df) => todo!("v_at table lookups"),
      _ => Err("type"),
    },
    K::SymbolArray(ss) => match l.clone() {
      K::Dictionary(d) => {
        Ok(K::List(
          // ss.categorical()
          ss.utf8()
            .unwrap()
            .into_iter()
            .map(|s| {
              if d.contains_key(s.unwrap().into()) {
                d.get(s.unwrap().into()).unwrap().clone()
              } else {
                K::Nil // TODO Is this the same behaviour as ngn/k and k9?
              }
            })
            .collect::<Vec<K>>(),
        ))
      }
      K::Table(_df) => todo!("v_at table lookups"),
      _ => Err("type"),
    },
    _ => todo!("v_at"),
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

pub fn v_iota(r: K) -> Result<K, &'static str> {
  debug!("v_iota");
  match r {
    K::Int(Some(i)) => Ok(K::IntArray(arr![(0..i).collect::<Vec<i64>>()])),
    _ => todo!("v_iota variants"),
  }
}
pub fn v_sum(x: K) -> Result<K, &'static str> {
  match x {
    K::BoolArray(a) => Ok(K::Int(a.sum())),
    K::IntArray(a) => Ok(K::Int(a.sum())),
    K::FloatArray(a) => Ok(K::Float(a.sum().unwrap_or(f64::NAN))),
    _ => todo!("other types of K"),
  }
}
pub fn v_d_sum(l: K, r: K) -> Result<K, &'static str> { Ok(l + v_sum(r).unwrap()) }

pub fn v_d_bang(l: K, r: K) -> Result<K, &'static str> {
  match l {
    K::SymbolArray(_) | K::Symbol(_) => v_makedict(l, r),
    _ => v_mod(l, r),
  }
}

pub fn v_each(_env: &mut Env, _v: KW, _x: K) -> Result<K, &'static str> { todo!("each") }
pub fn v_d_each(_env: &mut Env, _v: KW, _x: K, _y: K) -> Result<K, &'static str> { todo!("each") }
pub fn v_fold(env: &mut Env, v: KW, x: K) -> Result<K, &'static str> {
  // split into list, then reduce
  if let KW::Verb { name } = v {
    k_to_vec(x).and_then(|v| {
      let r = v.iter().cloned().reduce(|x, y| {
        apply_primitive(env, &name, Some(KW::Noun(x.clone())), KW::Noun(y.clone()))
          .unwrap()
          .unwrap_noun()
      });
      match r {
        Some(k) => Ok(k.clone()),
        None => Err("TODO not sure what this error case is"),
      }
    })
  } else {
    Err("type")
  }
}
pub fn v_d_fold(env: &mut Env, v: KW, x: K, y: K) -> Result<K, &'static str> {
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
    (K::Int(Some(2i64)), KW::Noun(K::Symbol(s))) => {
      let p = Path::new(&s);
      if p.exists() {
        match Path::new(&s).extension() {
          Some(e) => {
            if e == "csv" {
              Ok(KW::Noun(K::Table(
                CsvReader::from_path(p).unwrap().has_header(true).finish().unwrap(),
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
