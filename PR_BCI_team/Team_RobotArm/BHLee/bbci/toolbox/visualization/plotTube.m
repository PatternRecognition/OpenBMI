function hp = plotTube(tube, xx, col)
%plotTube(tube, xx, <col>)

versionString= version;
if versionString(1)<'6',
  plotTubeNoFaceAlpha(tube, xx);
  return;
end

nClasses= size(tube, 3);
if ~exist('col','var'),
  if size(tube,3)==1,
    col= {[0.8 0 0.8]};
  else
    col= {[0.8 0 0], [0 0.6 0], [0 0 1], [1 0.8 0], [0.8 0 0.8]};
  end
end
if ~exist('xx', 'var'),
  xx= 1:size(tube,1);
end
xx= xx(:);

clf;
hold on;
nShadings= (size(tube,2)-3)/2;
faceAlpha= linspace(0.2, 0.4, nShadings);
for si= 1:nShadings,
  for ci= 1:nClasses,
    hp= patch([xx; flipud(xx)], ...
              [tube(:,1+si,ci); flipud(tube(:,size(tube,2)-si,ci))], ...
              col{ci});
    set(hp, 'faceAlpha',faceAlpha(si), 'lineStyle','none');
  end
end
for ci= 1:nClasses,
  hp(ci)= plot(xx, tube(:,2+nShadings,ci));
  set(hp(ci), 'color',col{ci}, 'lineWidth',5);
end
for ci= 1:nClasses,                   %% separate loops for correct legend
  hpp= plot(xx, tube(:,2+nShadings,ci));
  set(hpp, 'color','k');
%  hp= plot(xx, tube(:,[1 5],ci));
%  set(hp, 'color',col{ci}, 'lineStyle','--');
end

xLim= xx([1 end])+[-1; 1]*0.02*(xx(end)-xx(1));
set(gca, 'xLim',xLim);
%h= line(xLim, [0 0]);
%set(h, 'color','k', 'lineStyle',':');
grid on;
hold off;
