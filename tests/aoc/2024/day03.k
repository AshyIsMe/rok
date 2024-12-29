i:,/0:"day03.txt"
m:{i[x + !12]}'&"m"=i /possible matches
m:m[&{&/"mul("=4#x}'m] /valid starts
m:m[&~^{x?")"}'m] /valid ends
m:{((4_x)?")")#(4_x)}'m /within parens
m:m[&{~^&/"0123456789,"?x}'m] /valid inner
+/{*/.'","\x}'m /part1 = 173419328

/part2
/find valid mul(x,y) indices
/filter for only indices within validrange blocks

doi:0,{d@&(&/"do()"=)'x@((!4)+)'d:&"d"=x} i /do indices
donti:{d@&(&/"don't()"=)'x@((!7)+)'d:&"d"=x} i /dont indices
vr:+{x,donti@*&donti>x}'doi /valid ranges

vs:{&/"mul("=4#x} /valid start
ve:{~^x?")"} /valid end
vi:{~^&/"0123456789,"?4_(x?")")#x} /valid inner
v:{vs[x]&ve[x]&vi[x]} /valid mul(x,y) instruction

mi:&"m"=i /m indices
mi:mi[&v'{i[x+!12]}'mi] /valid m indices

/WRONG mi:mi@&{x<(vr@1)@*&x<*vr}'mi /valid range m indices

/each mi is valid if:
/ doi to it's left is greater than donti to it's left
mi:mi@&>/(+{doi@*|&doi<x}'mi),+{donti@*|&donti<x}'mi /valid range m indices

e:{s:i[x+!12];","\4_(s?")")#s} /extract mul(x,y)
p:{s:i[x+!12];.'","\4_(s?")")#s} /parse mul(x,y)

+/*/+p'mi /part2 = 90669332
