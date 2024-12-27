i:,/0:"day03.txt"
m:{i[x + !12]}'&"m"=i /possible matches
m:m[&{&/"mul("=4#x}'m] /valid starts
m:m[&~^{x?")"}'m] /valid ends
m:{((4_x)?")")#(4_x)}'m /within parens
m:m[&{~^&/"0123456789,"?x}'m] /valid inner
+/{*/.'","\x}'m /part1 = 173419328
