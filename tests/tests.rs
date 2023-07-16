use polars::prelude::*;
use rok::*;

use rok::KW::*;

#[test]
fn test_1() {
    assert_eq!(scan("1").unwrap(), vec![Noun(K::Bool(1u8))]);
    assert_eq!(scan("2").unwrap(), vec![Noun(K::Int(Some(2)))]);
    assert_eq!(scan("3.14").unwrap(), vec![Noun(K::Float(3.14))]);
    assert_eq!(
        scan("1 0 1 0 1").unwrap(),
        vec![Noun(K::BoolArray(Series::new("", [1, 0, 1, 0, 1u8])))]
    );
    assert_eq!(scan("1 2 3").unwrap(), vec![Noun(K::IntArray(Series::new("", [1, 2, 3i64])))]);
    assert_eq!(
        scan("1 2 3.14").unwrap(),
        vec![Noun(K::FloatArray(Series::new("", [1., 2., 3.14])))]
    );
}

#[test]
fn test_2() {
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
        Noun(K::IntArray(Series::new("", [3, 2, 3, 2, 3])))
    );

    // assert_eq!(eval(scan("1 2 3 + 4 5 6").unwrap()).unwrap(), Noun(K::IntArray(Series::new("", [5, 7, 9]))));
}
