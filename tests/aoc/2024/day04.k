i:0:"day04sample.txt"
i:0:"day04.txt"
xmas:{I@&(&/"XMAS"=x@)'((!4)+)'I:&"X"=x} /indices of XMAS in string x
rot:{y(#y)!x+!#y}

l2r:#,/xmas'i /left2right count
r2l:#,/xmas'|'i /right2left count
t2b:#,/xmas'(|+:)i /top2bottom count
b2t:#,/xmas'|'(|+:)i /bottom2top count


/TODO diagonals aren't right

d:#,/xmas'+{rot[*x;1_x]}'(!#i),'i /diagonal
dr:#,/xmas'|'+{rot[*x;1_x]}'(!#i),'i /diagonal reversed

od:#,/xmas'+{rot[*x;1_x]}'(!#i),'|'i /other diagonal
odr:#,/xmas'|'+{rot[*x;1_x]}'(!#i),'|'i /other diagonal reversed

+/ l2r,r2l,t2b,b2t,od,odr /part1 sample = 18


+/ l2r,r2l,t2b,b2t,d,dr,od,odr /part1 = 1723 INCORRECT too low
