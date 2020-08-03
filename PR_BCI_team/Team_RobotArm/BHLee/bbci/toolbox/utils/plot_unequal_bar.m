function plot_unequal_bar(x,color,dist,name)
% PLOT_UNEQUAL_BAR plots a bar with blocks of different length
%
% usage: 
%    plot_unequal_bar(x)
% 
% input: 
%    x       a cell array with length = number of blocks and arrays
%            as entries  (zeros on the left and right were ignored)
%            or a matrix with blocks written in rows (zeros on the
%            left and right were ignored)
%    dist    is a scaling factor, which describes how many bar are
%            between the blocks
%    name    a list of names given to the blocks (for XTick)
%
% PLOTS THE BAR PLOT

if ~exist('color','var') | isempty(color)
  color = {'r','g','b','c','m','y'};
end

if ~exist('dist','var') | isempty(dist)
  dist = 1;
end


if iscell(x)
  xx = x{1};
  for i = 2:length(x)
    xxx = x{i};
    if size(xx,2)<length(xxx)
      xx = cat(2,xx,zeros(i-1,length(xxx)-size(xx,2)));
    elseif size(xx,2)>length(xxx)
      xxx = cat(2,xxx,zeros(1,size(xx,2)-length(xxx)));
    end
    
    xx = cat(1,xx,xxx);
  end
  x = xx;

end

if ~exist('name','var') | isempty(name)
  name = num2cell(1:size(x,1));
end


ind = zeros(size(x,1),2);

for i = 1:size(x,1);
  ro = x(i,:);
  in = find(ro~=0);
  ind(i,:) = [min(in),max(in)];
end


xx = [];
gr = [];
mx = [];
for i = 1:size(x,1)
  ro = x(i,ind(i,1):ind(i,2));
  xx = [xx,ro,zeros(1,dist)];
  mx = [mx,length(gr)+1+0.5*(ind(i,2)-ind(i,1))];
  gr = [gr,ind(i,1):ind(i,2),zeros(1,dist)];
  
end

xx(end-dist+1:end) = [];
gr(end-dist+1:end) = [];

hold on;
for i = 1:max(gr)
  xxx = zeros(1,length(xx));
  in = find(gr==i);
  xxx(in) = xx(in);
  h = bar(xxx);
  set(h,'FaceColor',color{i});
end

% $$$ 
% $$$ ch = get(gca,'Children');
% $$$ 
% $$$ for i = 1:length(ch)
% $$$   ab{i} = get(ch(i),'Vertices');
% $$$ end
% $$$ 
% $$$ 
% $$$ begin = ab{end}(1,1);
% $$$ step1 = ab{end}(2,1)-ab{end}(1,1);
% $$$ step2 = ab{end}(4,1)-ab{end}(3,1);
% $$$ 
% $$$ step4 = step1+step2;
% $$$ step3 = ab{end}(6,1)-ab{1}(5,1);
% $$$ 
% $$$ step3 = step3*dist;
% $$$ 
% $$$ pos = zeros(size(x));
% $$$ for i = 1:size(ind,1)
% $$$   pos(i,1:ind(i,1)-1) = begin;
% $$$   for j = ind(i,1):ind(i,2)-1
% $$$     pos(i,j) = begin;
% $$$     begin = begin+step4;
% $$$   end
% $$$   pos(i,ind(i,2):end) = begin;
% $$$   begin = begin+step3;
% $$$ end
% $$$ 
% $$$ for i = 1:length(ab)
% $$$   for j = 1:size(ind,1)
% $$$     po = pos(j,length(ab)+1-i);
% $$$     ab{i}(j*5-4,1) = po;
% $$$     ab{i}(j*5-3,1) = po+step1;
% $$$     ab{i}(j*5-2,1) = po+step1;
% $$$     ab{i}(j*5-1,1) = po+step4;
% $$$     ab{i}(j*5,1) = po+step4;
% $$$   end
% $$$   ab{i}(end,1)= ab{i}(end-1,1);
% $$$ end
% $$$ 
% $$$ for i = 1:length(ch)
% $$$   set(ch(i),'Vertices',ab{i});
% $$$ end
% $$$ 

set(gca,'XTick',mx);
set(gca,'XTickLabel',name);
