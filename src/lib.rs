// AA TODO: Initially I thought all K Arrays should be arrow arrays...
// but what if they were polars.Series and polars.DataFrames!?
// Then we'd get fast csv/parquet/json for "free".

#[derive(Clone)]
pub enum K {
    Bool(u8),
    Int(Option<i64>), //blech option here
    Float(f64),
    Char(char),
    //Symbol(i64), // index into global Symbols array?
    // List(Vec<K>), // USE arrow instead! KIntArray(arrow::Int64Array)
    //Dictionary{ vals: Vec<K>, keys: Vec<K> },
    // Function{ body, args, curry, env }
    // View{ value, r, cache, depends->val }
    // NameRef { name, l(index?), r(assignment), global? }
    Verb {
        name: String,
        l: Option<Box<K>>,
        r: Box<K>,
        curry: Option<Vec<K>>,
    },
    // Adverb { name, l(?), verb, r }
    Nil,
    // Cond { body: Vec< Vec<K> > } //list of expressions...
    //Quote(Box<K>)
}

pub fn scan(code: &str) -> Result<Vec<K>, &'static str> {
    let mut words = vec![];
    let mut skip: usize = 0;
    for (i, c) in code.char_indices() {
        if skip > 0 {
            skip -= 1;
            continue;
        }
        match c {
            '0'..='9' | '-' => {
                let (j, k) = scan_number(&code[i..])?;
                words.push(k);
                skip = j;
            }
            ':' | '+' | '-' | '*' | '%' | '!' | '&' | '|' | '<' | '>' | '=' | '~' | ',' | '^'
            | '#' | '_' | '$' | '?' | '@' | '.' => return Err("TODO scan primitive"),
            _ => return Err("TODO"),
        };
    }
    Ok(words.into())
}

pub fn scan_number(code: &str) -> Result<(usize, K), &'static str> {
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
            1 => Ok((l, nums[0].clone())),
            _ => Ok((l, promote_num(nums).unwrap())),
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
    if nums
        .iter()
        .any(|k| if let K::Float(f) = k { true } else { false })
    {
        todo!("promote float")
    } else if nums
        .iter()
        .any(|k| if let K::Int(i) = k { true } else { false })
    {
        todo!("promote int")
    } else if nums
        .iter()
        .all(|k| if let K::Bool(i) = k { true } else { false })
    {
        let bs: Vec<bool> = nums
            .iter()
            .map(|k| match k {
                K::Bool(0) => false,
                K::Bool(1) => true,
                _ => panic!("invalid bool"),
            })
            .collect();

        todo!("here")
    } else {
        Err("invalid nums")
    }
}
