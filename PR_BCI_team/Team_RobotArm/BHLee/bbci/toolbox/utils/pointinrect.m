function idx= pointinrect(p, rect)
%idx= pointinrect(p, rect)
% 
% IN  p    - point as [x y] vector
%     rect - rectangulars specified as [nRect 4] matrix,
%            with each row defining a rectangular by
%            [left bottom right top]
%
% OUT idx  - indices of the rectangulars that contain point p

idx= find(p(1)>=rect(:,1) & p(1)<=rect(:,3) & ...
          p(2)>=rect(:,2) & p(2)<=rect(:,4));
