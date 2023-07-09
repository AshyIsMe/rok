use std::io::{self, Write};


#[derive(Clone)]
pub enum K {
    Bool(i8),
    Int(i64),
    Float(f64),
    Char(char),
    Symbol(i64), // index into global Symbols array?
    List(Vec<K>),
    Dictionary{ vals: Vec<K>, keys: Vec<K> },
    // Function{ body, args, curry, env }
    // View{ value, r, cache, depends->val }
    // NameRef { name, l(index?), r(assignment), global? }
    Verb{ name: String, l: Option<Box<K>>, r: Box<K>, curry: Option<Vec<K>> },
    // Adverb { name, l(?), verb, r }
    Nil,
    // Cond { body: Vec< Vec<K> > } //list of expressions...
    Quote(Box<K>)
}

fn main() {
    println!("rok v0.000");

    let mut buffer = String::new();
    let stdin = io::stdin();
    let mut stdout = io::stdout();
    loop {
    stdout.write_all(b" ").unwrap(); //prompt
    stdout.flush().unwrap();
    stdin.read_line(&mut buffer).unwrap();

    stdout.write_all(b"TODO: Actual eval!\n").unwrap(); //prompt

    buffer.truncate(0);
    }

}
