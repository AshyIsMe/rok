i:0:"day04.txt"
i:0:"day04sample.txt"
xmas:{I@&(&/"XMAS"=x@)'((!4)+)'I:&"X"=x} /indices of XMAS in string x

l2r:#,/xmas'i /left2right count
r2l:#,/xmas'|'i /right2left count
t2b:#,/xmas'(|+:)i /top2bottom count
b2t:#,/xmas'|'(|+:)i /bottom2top count
/TODO diagonals
