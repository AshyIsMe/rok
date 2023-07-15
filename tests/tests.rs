use polars::prelude::*;
use rok::*;

#[test]
fn test_1() {
    assert_eq!(scan("1").unwrap(), vec![K::Bool(1u8)]);
    assert_eq!(scan("2").unwrap(), vec![K::Int(Some(2))]);
    assert_eq!(scan("3.14").unwrap(), vec![K::Float(3.14)]);
    assert_eq!(scan("1 0 1 0 1").unwrap(), vec![K::BoolArray(Series::new("", [1, 0, 1, 0, 1u8]))]);
    assert_eq!(scan("1 2 3").unwrap(), vec![K::IntArray(Series::new("", [1, 2, 3i64]))]);
    assert_eq!(scan("1 2 3.14").unwrap(), vec![K::FloatArray(Series::new("", [1., 2., 3.14]))]);
}

#[test]
fn test_2() {
    assert_eq!(
        scan("2 + 2").unwrap(),
        vec![K::Int(Some(2)), K::Verb { name: "+".to_string() }, K::Int(Some(2))]
    );
    assert_eq!(eval(scan("2 + 2").unwrap()).unwrap(), K::Int(Some(4)));

    assert_eq!(
        scan("1.0 + 2.0").unwrap(),
        vec![K::Float(1.0), K::Verb { name: "+".to_string() }, K::Float(2.0)]
    );
    assert_eq!(eval(scan("1.0 + 2.0").unwrap()).unwrap(), K::Float(3.0));

    assert_eq!(eval(scan("1 0 1 0 1 + 2").unwrap()).unwrap(), K::IntArray(Series::new("", [3, 2, 3, 2, 3])));
}
