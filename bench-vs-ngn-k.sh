#!/usr/bin/env sh

# ngn/k
mkdir -p ngn
git clone https://codeberg.org/ngn/k ./ngn/k || true
pushd ./ngn/k
git pull
make k
popd


# rok
cargo build --release

# 8GB of i64
B="\\\t #!1000000000"
echo "ngn/k time: $B"
echo "$B" | ./ngn/k/k
echo "rok time: $B"
echo "$B" | ./target/release/rok

# 8GB of i64
B="\\\t #2*!1000000000"
echo "ngn/k time: $B"
echo "$B" | ./ngn/k/k
echo "rok time: $B"
echo "$B" | ./target/release/rok

B="\\\t +/2*!1000000000"
echo "ngn/k time: $B"
echo "$B" | ./ngn/k/k
echo "rok time: $B"
echo "$B" | ./target/release/rok
