use polars::prelude::*;
use std::ops;

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
    //Symbol(i64), // index into global Symbols array?
    BoolArray(Series),
    IntArray(Series), // ints are nullable so have to be a series
    FloatArray(Series),
    Nil, // Is Nil a noun?
         //Dictionary{ vals: Vec<K>, keys: Vec<K> },
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
}

pub fn apply_primitive(v: &str, l: Option<KW>, r: KW) -> Result<KW, &'static str> {
    match v {
        "+" => match (l, r) {
            (Some(KW::Noun(l)), KW::Noun(r)) => Ok(KW::Noun(v_plus(l, r).unwrap())),
            _ => todo!("monad +"),
        },
        "-" => match (l, r) {
            (Some(KW::Noun(l)), KW::Noun(r)) => Ok(KW::Noun(v_minus(l, r).unwrap())),
            _ => todo!("monad -"),
        },
        "*" => match (l, r) {
            (Some(KW::Noun(l)), KW::Noun(r)) => Ok(KW::Noun(v_times(l, r).unwrap())),
            _ => todo!("monad *"),
        },
        "%" => match (l, r) {
            (Some(KW::Noun(l)), KW::Noun(r)) => Ok(KW::Noun(v_divide(l, r).unwrap())),
            _ => todo!("monad %"),
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
    ($op:tt, $self:ident, $r:ident) => {
        match ($self.clone(), $r) {
            // There must be an easier way... What's a macro???
            (K::Bool(l), K::Bool(r)) => K::Int(Some(l as i64 $op r as i64)),
            (K::Bool(l), K::Int(Some(r))) => K::Int(Some(l as i64 $op r)),
            (K::Int(Some(l)), K::Bool(r)) => K::Int(Some(l $op r as i64)),
            (K::Bool(l), K::Float(r)) => K::Float(l as f64 $op r),
            (K::Float(l), K::Bool(r)) => K::Float(l $op r as f64),

            (K::BoolArray(l), K::BoolArray(r)) => K::IntArray(l $op r),
            (K::BoolArray(_l), K::Int(Some(r))) => match b2i($self) {
                K::IntArray(l) => K::IntArray(l $op r),
                _ => panic!("impossible"),
            },
            (K::Int(Some(l)), K::BoolArray(r)) => K::IntArray(r $op l),
            (K::BoolArray(l), K::Float(r)) => K::FloatArray(l $op r),
            (K::Float(l), K::BoolArray(r)) => K::FloatArray(r $op l),

            (K::Int(Some(l)), K::Int(Some(r))) => K::Int(Some(l $op r)),
            (K::Float(l), K::Float(r)) => K::Float(l $op r),
            _ => todo!("various $op pairs - LOTS MORE to do still: dicts/tables/etc"),
        }
    };
}

impl ops::Add for K {
    type Output = Self;
    fn add(self, r: Self) -> Self::Output { impl_op!(+, self, r) }
}
impl ops::Sub for K {
    type Output = Self;
    fn sub(self, r: Self) -> Self::Output { impl_op!(-, self, r) }
}
impl ops::Mul for K {
    type Output = Self;
    fn mul(self, r: Self) -> Self::Output { impl_op!(*, self, r) }
}
impl ops::Div for K {
    type Output = Self;
    fn div(self, r: Self) -> Self::Output { impl_op!(/, self, r) }
}

pub fn v_plus(l: K, r: K) -> Result<K, &'static str> { Ok(l + r) }
pub fn v_minus(l: K, r: K) -> Result<K, &'static str> { Ok(l - r) }
pub fn v_times(l: K, r: K) -> Result<K, &'static str> { Ok(l * r) }
pub fn v_divide(l: K, r: K) -> Result<K, &'static str> { Ok(l / r) }

pub fn eval(ast: Vec<KW>) -> Result<KW, &'static str> {
    match &ast[..] {
        [] => Err("Noop not implemented"),
        [KW::Noun(_)] => Ok(ast[0].clone()),
        [KW::Verb { name: _ }, KW::Noun(_)] => todo!("monad"),
        [KW::Noun(_), KW::Verb { name }, KW::Noun(_)] => {
            let r = apply_primitive(name, Some(ast[0].clone()), ast[2].clone()).unwrap();
            let mut new_ast: Vec<KW> = vec![]; // TODO: shorten this, concat!() macro?
            new_ast.extend(ast[..ast.len()-3].iter().cloned());
            new_ast.append(&mut vec![r]);
            eval(new_ast)
        }
        [_, _, _] => Err("syntax error"),
        [_, ..] => {
            let r = eval(ast[ast.len() - 3..].to_vec()).unwrap();
            let mut new_ast: Vec<KW> = vec![]; // TODO: shorten this, concat!() macro?
            new_ast.extend(ast[..ast.len()-3].iter().cloned());
            new_ast.append(&mut vec![r]);
            eval(new_ast)
        },
    }
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
            '0'..='9' | '-' => {
                if let Ok((j, k)) = scan_number(&code[i..]) {
                    words.push(k);
                    skip = j;
                } else {
                    words.push(KW::Verb { name: c.to_string() })
                }
            }
            ':' | '+' | '*' | '%' | '!' | '&' | '|' | '<' | '>' | '=' | '~' | ',' | '^' | '#'
            | '_' | '$' | '?' | '@' | '.' => words.push(KW::Verb { name: c.to_string() }),
            ' ' | '\t' | '\n' => continue,
            _ => return Err("TODO"),
        };
    }
    Ok(words.into())
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
    if nums.iter().any(|k| if let K::Float(_f) = k { true } else { false }) {
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
    } else if nums.iter().any(|k| if let K::Int(_i) = k { true } else { false }) {
        let ia: Vec<Option<i64>> = nums
            .iter()
            .map(|k| match k {
                K::Bool(i) => Some(*i as i64),
                K::Int(i) => *i,
                _ => panic!("invalid int"),
            })
            .collect();

        Ok(K::IntArray(Series::new("", ia)))
    } else if nums.iter().all(|k| if let K::Bool(_i) = k { true } else { false }) {
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
