+/ (safe:{d:1_-':x; ad:(d|-d); ((&/d<0)|(&/d>0)) & (&/ad<4) & (&/ad>0)})' i:.'0:"day02.txt" /part1 = 516

+/(safe2:{|/safe'x(&'(,5#1),~=5)})'i /part2 = 689 INCORRECT
/689 incorrect - right answer for someone else
/794 too high

+/(safe2:{$[safe x;1;1=+/safe'x(&'~=5)]})'i /part2 = 613 incorrect


1 0 0 0 0 1 ~ safe'(7 6 4 2 1; 1 2 7 8 9; 9 7 6 2 1; 1 3 2 4 5; 8 6 4 4 1; 1 3 6 7 9) / test1 correct
1 0 0 1 1 1 ~ safe2'(7 6 4 2 1; 1 2 7 8 9; 9 7 6 2 1; 1 3 2 4 5; 8 6 4 4 1; 1 3 6 7 9) / test2 correct



/ attempt 2 ==========================

safeinc:{*/(4>d) & 0<d:1_-':x}
safedec:{*/(-4<d) & 0>d:1_-':x}
+/ (safe:{(safeinc x)|safedec x})'i /part1 = 516

+/(safe2:{|/safe'x(&'(,5#1),~=5)})'i /part2 689 INCORRECT
