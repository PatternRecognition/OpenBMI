function showClassDistr(out, le, ri, xx, Col)
% showClassDistr(out, le, ri, <xx, Col>)

if ~exist('Col', 'var'),
  Col= [1 0.09 0.13; 0 0.64 0.54];
end
if ~exist('xx','var'), xx=20; end

if length(xx)==1,
  mm= max(max(out), -min(out));
  xx= linspace(-mm, mm, xx);
end

nn_le= hist(out(le), xx);
hb= bar(xx, nn_le);
set(hb, 'faceColor', Col(1,:));
nn_ri= hist(out(ri), xx);
hold on;
hb= bar(xx, nn_ri);
set(hb, 'faceColor', Col(2,:));
tofront= find(nn_le<nn_ri);
hb= bar(xx(tofront), nn_le(tofront));
set(hb, 'faceColor', Col(1,:));
hold off;
axis tight;
yLim= get(gca, 'yLim');
yLim(2)= yLim(2) + 0.05*diff(yLim);
set(gca, 'yLim', yLim);
