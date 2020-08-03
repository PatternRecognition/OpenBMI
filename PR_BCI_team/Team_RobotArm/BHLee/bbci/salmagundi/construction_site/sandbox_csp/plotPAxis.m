function plotPAxis(U, D, cent, col)
% plotPAxis(U, D, <center, col>)

if nargin<3, cent=[0 0]; end
if nargin<4, col='y'; end

for d= 1:2,
  xd= sqrt(D(d,d))*U(1,d);
  yd= sqrt(D(d,d))*U(2,d);
  line(cent(1)+[-xd xd], cent(2)+[-yd yd], 'lineWidth', 3, 'color', col);
  hl(d)= line(cent(1)+[-xd xd], cent(2)+[-yd yd], 'color', 'k');
end
moveObjectForth(hl(1));
