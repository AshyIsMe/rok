use rok::*;

#[test]
fn test_1() {
    let k = K::Bool(1u8);
    let ast = scan("1").unwrap();
    let ast = scan("2").unwrap();
    let ast = scan("3.14").unwrap();
    assert!(true);
}

fn test_2() {
    let k = K::Bool(1u8);
    let ast = scan("2 + 2").unwrap();
    assert!(true);
}