function animate_fb_pseudo_brainrace(package,varargin);
% matlab feedback for brainrace two player....

persistent fig no1 hax hb

if ischar(package) 
  fig = figure;
  set(fig,'CloseRequestFcn','global run;closereq;run=0;');
  set(fig,'MenuBar','none');
  hb(1) = patch([-1 -1 0 0], [-1 1 1 -1], 'r');
  hb(2) = patch([0 0 1 1], [-1 1 1 -1], 'b');
 % hax = axes('position',[-1 -1 1 1],'ylim',[-1 1],'box','on');
  %set(gcf,'DoubleBuffer','on');
  set(gca,'ylim',[-1,1])
  l = line([-1 1],[0 0]);
  set(l,'color','k');
  
  no1 = 0;
  axis normal
  axis off
 drawnow

else
  if package(1)<0, package(1)=256+package(1);end
  if package(1)-no1>1
    fprintf('You have lost %i packages\n',package(1)-no1-1);
  end
  no1 = package(1);
  set(hb(1), 'ydata', [-1 package(2) package(2) -1]);
  set(hb(2), 'ydata', [-1 package(3) package(3) -1]);
  drawnow;
end
