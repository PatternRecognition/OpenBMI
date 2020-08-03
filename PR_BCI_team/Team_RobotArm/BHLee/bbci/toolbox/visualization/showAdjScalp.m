function showAdjScalp(mnt, adj, varargin)
% showAdjScalp(mnt, adj, <opt>)
%
% IN:  mnt           struct with field clab and pos_3d
%      adj           adjacency matrix to visualize (is assumed to be symmetric).
%      opt           options struct with possible fields:
%         SHOW_LABELS   plot labels on electrode patches
%         InnerCircle   plot inner circle
%         ElectrodeCircle  plot circle around each electrode
%         ElectrodeSize    size of each electrode in the plot.
%      
% recommendation: save the resulting figure with paperSize [35 20]
%                 (i.e. saveFigure('impedancefig', [35 20]))

% kraulem 08/04
if ~isempty(varargin)
  opt = varargin{1};
else
  opt = struct('SHOW_LABELS',1);
end
opt = set_defaults(opt,'SHOW_LABELS',1,...
                       'InnerCircle',0,...
                       'ElectrodeCircle',0,...
                       'ElectrodeSize',0.05,...
                       'FontSize',8);


mnt = projectElectrodePositions(mnt.clab);
mnt_index = 1:length(mnt.clab);
mnt.x = mnt.x(mnt_index);
mnt.y = mnt.y(mnt_index);
mnt.pos_3d = mnt.pos_3d(:,mnt_index);
mnt.clab = {mnt.clab{mnt_index}};

clf;
plotScalp(adj, mnt, opt);
return




function fig = plotScalp(adj, mnt, opt)
% plots the information from the array adj 

dispChans= find(~isnan(mnt.x));

xe= mnt.x(dispChans);
ye= mnt.y(dispChans);

r= opt.ElectrodeSize;
%inner circle:
T= linspace(0, 2*pi, 360);
xx= cos(T)/1.3;
yy= sin(T)/1.3;
if opt.InnerCircle
  plot(xx, yy, 'k');
end

T= linspace(0, 2*pi, 18);
disc_x= r*cos(T); 
disc_y= r*sin(T);

hold on;
for ia = 1:size(adj,1)
  for ib = 1:size(adj,2)
    if adj(ia,ib)
      l = line(xe([ia ib]),ye([ia ib]));
      set(l,'Color','k');
    end
  end
end
for ic= 1:length(dispChans),
  p =  patch(xe(ic)+disc_x, ye(ic)+disc_y, 'w');
  if ~opt.ElectrodeCircle
    set(p, 'EdgeColor', 'none');
  end
  set(p, 'FaceAlpha', 1);
end
%caxis(COL_AX);
%c = colormap(hsv(round((COL_AX(2)-COL_AX(1))*3.5)));
%c = c((COL_AX(2)-COL_AX(1)):(-1):1, :);
%colormap(c);


T= linspace(0, 2*pi, 360);
xx= cos(T);
yy= sin(T);
plot(xx, yy, 'k');
nose= [1 1.1 1];
nosi= [86 90 94]+1;
plot(nose.*xx(nosi), nose.*yy(nosi), 'k');

if opt.SHOW_LABELS,
  labs= {mnt.clab{dispChans}};
  for il= 1:length(labs),
    strLen(il)= length(labs{il});
    %labs{il}=[labs{il}, sprintf('\n%d', impedances(il))]; 
  end
  h= text(xe, ye, labs);
  set(h, 'fontSize',opt.FontSize, 'horizontalAlignment','center', 'verticalAlignment','middle');
  iLong= find(strLen>3);
  set(h(iLong),'fontSize',opt.FontSize-1);
end

hold off;



axis('xy', 'tight', 'equal', 'tight','off');


return