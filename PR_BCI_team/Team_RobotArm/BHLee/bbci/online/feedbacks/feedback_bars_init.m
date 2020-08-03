function [handle] = feedback_bars_init(fig,fb_opt);

nClasses = length(fb_opt.nClasses);


xxx = (0.9-0.1*nClasses)/(nClasses+1);
xxx = (1:nClasses)*(xxx+0.1)-0.05;
barColor = zeros(9,3,3);
for i = 1:3
  barColor(i,i,1) = 0.8;
  barColor(i,:,2) = 0.7;
  barColor(i,i,2:3) = 1;
  barColor(i+3,:,1) = 0.5;
  barColor(i+3,:,2:3) = 0.8;
  barColor(i+3,i,[1,3]) = 0;
  barColor(i+3,i,2) = 0.4;
  barColor(i+6,i,1) = 0.5;
  barColor(i+6,:,2) = 0.4;
  barColor(i+6,i,2:3) = 0.8;
  
end


hb = zeros(1,length(xxx));
hl = zeros(1,length(xxx));
ha = zeros(1,length(xxx));
hax = zeros(1,length(xxx));
  
%figure zeichnen
clf;
set(fig,'position', fb_opt.position);
for ii = 1:nClasses
  iii = fb_opt.classOrder(ii);
  set(fig,'DoubleBuffer','on');
  set(fig,'MenuBar','none');
  
  ha(iii) = axes('position', [xxx(ii) 0.05 0.1 0.7], 'box','on', ...
                   'xTick',[], 'yTick',[], 'yLim',fb_opt.yLim);
  xlabel(fb_opt.classes{iii});
  hb(iii)= patch([0 1 1 0], [0 0 0 0], barColor(iii,:,1));
  %   set(hb(iii),'EraseMode','xor');
  hl(iii)=line([0 1],[1 1]*fb_opt.threshold , 'color','k');
  hax(iii)= axes('position', [xxx(ii) 0.8 0.1 0.1], 'box','on', ...
                 'xTick',[], 'yTick',[]);
end 

handle = [ha,hb,hl,hax,fig,gcf];
