function [handle,voi,xLim,yLim] = feedback_voiture_init(fig,opt);

clf;
set(fig, 'Menubar','none', ... %'Resize','off', ...
         'position',opt.position);
set(fig,'color',[1 1 1], 'DoubleBuffer','on', ...
        'Pointer','custom', 'PointerShapeCData',ones(16)*NaN);
set(gca, 'position',[0.03 0.03 0.94 0.94], ...
         'XLim',[-1 1], 'YLim',[-1 1], 'color',opt.background);
axis equal
drawnow;
set(gca, 'xTick',[], 'yTick',[], 'box','on', ...
         'xLimMode','manual', 'yLimMode','manual');
xLim= get(gca, 'xLim');
yLim= get(gca, 'yLim');

ht= zeros(1, opt.obstacles+1);
ht(1)= patch([0 0 0 0], [0 0 0 0], opt.target_color);
for ii= 1:opt.obstacles,
  ht(1+ii)= patch([0 0 0 0], [0 0 0 0], opt.obstacle_color);
end
set(ht, 'visible','off', 'edgeColor','none');
hw= text(xLim(2)-0.02*diff(xLim), yLim(2), '0:00', ...
         'fontSize',20, 'fontWeight','bold');
set(hw, 'horizontalAli','right', 'verticalAli','top');
set(hw, 'visible',opt.show_stopwatch);
hhm= text(xLim(1)+0.02*diff(xLim), yLim(2), '0:0', ...
          'fontSize',20, 'fontWeight','bold');
set(hhm, 'horizontalAli','left', 'verticalAli','top');
set(hhm, 'visible',opt.show_points);



%% initialize car data
voi.speed= 0;
voi.acc= opt.acceleration;
voi.fw= [mean(xLim); -0.75];
vv= [0; opt.carlength];
voi.rw= voi.fw - vv ;
voi.alpha= 0;

%% get size of display area
oldUnits= get(gca, 'unit');
set(gca, 'unit','pixel');
pos= get(gca, 'position');
set(gca, 'unit',oldUnits);
%% prepare graphic objects for the car
hold on;
switch(opt.car_type),
 case 'patch',
  hv= patch([0 0 0 0], [0 0 0 0], opt.car_color);
 case 'line',
  lineWidth= floor(opt.carwidth*pos(3)*0.7);
  hv= line([0 0], [0 0], 'color',opt.car_color, 'lineWidth',lineWidth);
 case 'marker', 
  markerSize= round(opt.carwidth*pos(3))*50/15;
  hv= plot([0 0], '.', 'color',opt.car_color, 'markerSize',markerSize);
end
hd= plot([0 0], [0 0], 'color',opt.direction_color, 'linewidth',2);
hf= plot([0 0], [0 0], 'color',opt.front_color, ...
         'lineWidth',opt.bumper_width);
hr= plot([0 0], [0 0], 'color',opt.rear_color, ...
         'lineWidth',opt.bumper_width);
hold off;
set(hd, 'visible',opt.show_direction);
set(hf, 'visible',opt.show_front);
set(hr, 'visible',opt.show_rear);

set([hv hd hf hr], 'eraseMode','xor');





handle = [hw,hhm,hv,hd,hf,hr,fig,gca,ht];

