function h= showMarker(mrk, varargin)
%showMarker(mrk, <OPT>)
%showMarker(mrk, idx, <sa2time, time0>)

if isempty(varargin) || ischar(varargin{1}) || isstruct(varargin{1}),
  opt= propertylist2struct(varargin{:});
  opt= set_defaults(opt, ...
                    'idx', 1:length(mrk.pos));
else
  opt.idx= varargin{1};
  varargin(1)= [];
  if ~isempty(varargin),
    opt.sa2time= varargin{1};
    varargin(1)= [];
  end
  if ~isempty(varargin),
    opt.time0= varargin{1};
    varargin(1)= [];
  end
  opt= merge_structs(opt, propertylist2struct(varargin{:}));
end
opt= set_defaults(opt, ...
                  'sa2time', 1000/mrk.fs, ...
                  'labelpos','bottom', ...
                  'time0', 0, ...
                  'triangle_width', 1/100, ...
                  'triangle_height', 1/40, ...
                  'shade_ival', [], ...
                  'shade_height', 0.3, ...
                  'shade_col', 0.3*[1 1 1]);
%                 'time0', mrk.pos(opt.idx(1))*opt.sa2time);

h= [];
k= 0;
for is= opt.idx,
  k= k+1;
  oldUnits= get(gca, 'units');
  set(gca, 'units','pixel');
  pos= get(gca, 'position');
  set(gca, 'units',oldUnits);
  x= mrk.pos(is)*opt.sa2time - opt.time0;
  yLim= get(gca, 'yLim');
%  pu= diff(yLim)/pos(4);
%  y= yLim(1) + [0.5 6]*pu;
%  h.line= line([x x], y);
%  set(h.line, 'color','r', 'linewidth',3);

  % draw vertical line at marker positions
  h.line(k)= line([x x], yLim, 'color','k', 'lineStyle','--');
  % add small red triangle
  xx= x + [-1 1]*diff(xlim)*opt.triangle_width;
  hh= diff(yLim)*opt.triangle_height;
  h.tri(k,1)= patch([xx mean(xx)], [yLim([1 1]) yLim(1)+hh], 'r');
  h.tri(k,2)= patch([xx mean(xx)], [yLim([2 2]) yLim(2)-hh], 'r');
  % optionally shade interval related to marker position
  if ~isempty(opt.shade_ival),
    xx= x + opt.shade_ival/1000;
    ic= min(find(mrk.y(:,is)), size(opt.shade_col,1));
    col= opt.shade_col(ic,:);
    for cc= 1:floor(yLim(2)),  %% Oh, boy!
      yy= cc + opt.shade_height*[-1 1];
      h.shade(k,cc)= patch(xx([1 2 2 1]), yy([1 1 2 2]), col);
    end
  end
  % add classname
  tag= '';
  if isfield(mrk, 'className'),
    tag= mrk.className{find(mrk.y(:,is))};
  elseif isfield(mrk, 'desc'),
    tag= mrk.desc{is};
  end
  if ~isempty(tag),
    switch(opt.labelpos),
     case 'top',
      h.text(k)= text(x, yLim(2), tag);
      set(h.text(k), 'horizontalAlignment','center', ...
              'verticalAlignment','top');
     case 'bottom',
      h.text(k)= text(x, yLim(1), tag);
      set(h.text(k), 'horizontalAlignment','center', ...
                     'verticalAlignment','bottom');
    end
  end
end
if ~isempty(opt.idx),
  if isfield(h, 'shade'),
    set(h.shade, 'EdgeColor','none');
    moveObjectBack(h.shade);
  end
  set(h.tri, 'EdgeColor','none');
end
