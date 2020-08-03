function colpale= col_makePale(col, v)

if nargin<2,
  v= 0.5;
end

col= rgb2hsv(col);
colpale= zeros(size(col));
for row= 1:size(col,1),
  if col(row,2)==0,
    colpale(row,:)= [col(row,1) col(row,2) 1-(v*(1-col(row,3)))];
  else
    colpale(row,:)= [col(row,1) v*col(row,2) col(row,3)];
  end
end
colpale= hsv2rgb(colpale);
