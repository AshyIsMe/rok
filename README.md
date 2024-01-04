# rok

A K interpreter written in rust.
Inspired by ok and ngn/k.

How far can we get by cheating and using polars for the array/tables data types?

oK.js is under 1k lines of javascript, can we do the same in under 1k lines of rust?

* https://www.arraycast.com/episodes/episode41-john-earnest
* https://github.com/JohnEarnest/ok
* https://codeberg.org/ngn/k
* https://github.com/razetime/ngn-k-tutorial/blob/main/01-intro.md
* https://k.miraheze.org/wiki/Main_Page


## debug run

```
RUST_LOG=rok=debug cargo run
```


## example

```
% cargo run
    Finished dev [unoptimized + debuginfo] target(s) in 0.08s
     Running `target/debug/rok`
rok 0.1.0
 1 2 3 + 4 5 6
5 7 9
 +/ 1 2 3 + 4 5 6
21
 !10
0 1 2 3 4 5 6 7 8 9
 1+!10
1 2 3 4 5 6 7 8 9 10
 */1+!10
3628800
 (1 2 3;"abc";3.14)
(1 2 3
 "abc"
 3.14)
 `a`b`c!(1 2 3;"abc";3.14)
a|1 2 3
b|"abc"
c|3.14
 (`a`b`c!(1 2 3;"abc";3.14); (!10))
(a|1 2 3
b|"abc"
c|3.14
 0 1 2 3 4 5 6 7 8 9 )
 (`a`b`c!(1 2 3;"abc";3.14); (!10);(4 5 6;"def"))
(a|1 2 3
b|"abc"
c|3.14
 0 1 2 3 4 5 6 7 8 9
 (4 5 6
 "def"))
 +`a`b`c!("abc";1 2 3; 10 20 30)
shape: (3, 3)
┌─────┬─────┬─────┐
│ a   ┆ b   ┆ c   │
│ --- ┆ --- ┆ --- │
│ str ┆ i64 ┆ i64 │
╞═════╪═════╪═════╡
│ 97  ┆ 1   ┆ 10  │
│ 98  ┆ 2   ┆ 20  │
│ 99  ┆ 3   ┆ 30  │
└─────┴─────┴─────┘

```
