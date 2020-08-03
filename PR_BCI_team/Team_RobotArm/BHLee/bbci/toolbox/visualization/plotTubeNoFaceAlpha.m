function hp= plotTubeNoFaceAlpha(tube, xx, varargin)
%plotTubeNoFaceAlpha(tube, <xx, col, showMinMax>)
%plotTubeNoFaceAlpha(tube, xx, opt>)
%
% IN  opt
%        .color            - hsv coded!
%        .showMinMax       - 0 (default) or 1
%        .xGrid, .yGrid    - 'on' (default) or 'off'
%        .median_lineWidth -

nClasses= size(tube, 3);
if ~exist('xx', 'var'),
  xx= 1:size(tube,1);
end
xx= xx(:);
if length(varargin)>=1,
  if isnumeric(varargin{1}),
    opt.color= varargin{1};
    if length(varargin)>=2,
      opt.showMinMax= varargin{2};
    
    end
  else
    opt= propertylist2struct(varargin{:});
  end
else
  opt = [];
end

opt= set_defaults(opt, ...
                  'showMinMax', 0, ...
                  'xGrid', 'on', ...
                  'yGrid', 'on', ...
                  'borderlines', 1, ...
                  'median_lineWidth', 5, ...
                  'median_lineWidth2', 1);
if ~isfield(opt, 'color') | isempty(opt.color),
  if size(tube,3)==1,
    opt.color= [0.85 1 0.9];
  else
    opt.color= [0 1 0.9;  0.28 1 0.7;  0.6 1 0.8;  0.13 1 1;  0.85 1 0.8];
  end
end
col= opt.color;

cla;
hold on;
nShadings= (size(tube,2)-3)/2;
for si= 1:nShadings,
  for ci= 1:nClasses,
    c= col(ci,:);
    if nShadings==1,
%      c(2)= 0.3;
%      c(3)= 0.85;
    else
      c(2)= 0.25 + (0.5-0.25)*((si-1)/(nShadings-1));
      c(3)= 1 + (c(3)-1)*((si-1)/(nShadings-1));
    end
    hp= patch([xx; flipud(xx)], ...
              [tube(:,1+si,ci); flipud(tube(:,size(tube,2)-si,ci))], ...
              hsv2rgb(c));
    set(hp, 'lineStyle','none');
  end
  if opt.borderlines,
    for ci= nClasses-1:-1:1,
      c= col(ci,:);
      if nShadings==1,
%        c(2)= 0.3;
%        c(3)= 0.85;
      else
        c(2)= 0.25 + (0.5-0.25)*((si-1)/(nShadings-1));
        c(3)= 1 + (c(3)-1)*((si-1)/(nShadings-1));
      end
      plot(xx, tube(:,1+si,ci), 'color',hsv2rgb(c), 'lineWidth',2);
      plot(xx, tube(:,size(tube,2)-si,ci), 'color',hsv2rgb(c), 'lineWidth',2);
    end
  end
end
for ci= 1:nClasses,
  hp(ci)= plot(xx, tube(:,2+nShadings,ci), ...
               'color',hsv2rgb(col(ci,:)), 'lineWidth',opt.median_lineWidth);
  plot(xx, tube(:,2+nShadings,ci), 'k', 'lineWidth',opt.median_lineWidth2);
  if opt.showMinMax,
    plot(xx, tube(:,[1 end],ci), ...
         'color',hsv2rgb(col(ci,:)), 'lineStyle','--');
  end
end
drawnow;

xLim= xx([1 end])+[-1; 1]*0.02*(xx(end)-xx(1));
set(gca, 'xLim',xLim);
grid_over_patches('xGrid',opt.xGrid, 'yGrid',opt.yGrid);
