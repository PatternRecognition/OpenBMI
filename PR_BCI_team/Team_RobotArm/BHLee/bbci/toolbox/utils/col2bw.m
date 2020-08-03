function [cols, hl]= col2bw(bwStyles, h, cols, hl)
%[cols, hl]= col2bw(bwStyles, <h>)
%
% bwStyles: cell array of line property defining strings
% h:        handle of figure

if ~exist('bwStyles', 'var') | isequal(bwStyles,1),
  bwStyles= {{'lineWidth',0.3}, ...
             {'lineWidth',1}, ...
             {'lineWidth',0.3, 'lineStyle','-.'}, ...
             {'lineWidth',1, 'lineStyle','-.'}, ...
             {'lineWidth',0.3, 'lineStyle','--'}, ...
             {'lineWidth',1, 'lineStyle','--'}, ...
             {'lineWidth',0.3, 'lineStyle',':'}, ...
             {'lineWidth',1, 'lineStyle',':'}};
elseif isequal(bwStyles,2),
  bwStyles= {{'lineStyle','-'}, ...
             {'lineStyle','-.'}, ...
             {'lineStyle','--'}, ...
             {'lineStyle',':'}};
elseif isequal(bwStyles,-1),
  bwStyles= {{'lineWidth',1}, ...
             {'lineWidth',0.3}, ...
             {'lineWidth',1, 'lineStyle','-.'}, ...
             {'lineWidth',0.3, 'lineStyle','-.'}};
elseif isequal(bwStyles,0),
  cols= [];
  hl= [];
  return;
end
if ~exist('h', 'var'), h=gcf; end
if ~exist('cols', 'var'), cols=[]; end
if ~exist('hl', 'var'), hl=zeros(0,3); end

versionString= version;
if strcmp(get(h,'tag'), 'legend') &  versionString(1)>='6', return; end;
%if strcmp(get(h,'type'), 'line') & strcmp(get(h,'marker'),'none'),
if strcmp(get(h,'type'), 'line'),
  lcol= get(h, 'color');
  if any(lcol~=lcol(1)*[1 1 1]),    %% leave uncolored lines unchanged
    ci= findRow(lcol, cols);
    if isempty(ci),
      cols= [cols; lcol];
      ci= size(cols, 1);
      if ci>length(bwStyles),
        error('more colors than styles encountered');
      end  
    end
    hl= [hl; h ci get(h,'lineWidth')];
    set(h, 'color','k', bwStyles{ci}{:});
  end
else
  hc= get(h, 'children');
  for h= hc',
    [cols,hl]= col2bw(bwStyles, h, cols, hl);
  end
end


function ri= findRow(row, mat)

nRows= size(mat,1);
ri= zeros(nRows, 1);
for ii= 1:nRows,
  ri(ii)= all(row==mat(ii,:));
end
ri= find(ri);
