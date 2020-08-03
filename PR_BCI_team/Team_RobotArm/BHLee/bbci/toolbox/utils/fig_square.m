function fig_square

oldUnit= get(gcf, 'Unit');
set(gcf, 'Unit', 'Centimeters');
pos= get(gcf, 'Position');
pos(3:4)= min(pos(3:4));
set(gcf, 'Position',pos);
set(gcf, 'Unit',oldUnit);
