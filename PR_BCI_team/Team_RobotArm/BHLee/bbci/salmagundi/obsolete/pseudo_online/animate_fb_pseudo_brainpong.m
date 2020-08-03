function animate_fb_pseudo_brainpong(package,varargin);
% matlab feedback for brainpong two player....

persistent fig width height fb_opt no1 no2

if ischar(package) 
  fig = figure;
  set(fig,'CloseRequestFcn','global run;closereq;run=0;');
  set(fig,'MenuBar','none');
  axes('position',[0 0 1 1]);
  if length(varargin)>0
    width = varargin{1};
  else
    width = 0.3;
  end
  fb_opt.racket1_x= [-width;width];
  fb_opt.racket1_y= [0.05 0.05]';
  fb_opt.h_racket1= ...
      line(fb_opt.racket1_x, fb_opt.racket1_y, ...
           'color','b', 'lineWidth',30);
  fb_opt.racket2_x= [-width;width];
  fb_opt.racket2_y= [0.95 0.95]';
  fb_opt.h_racket2= ...
      line(fb_opt.racket2_x, fb_opt.racket2_y, ...
           'color','b', 'lineWidth',30);

  no1 = 0;
  no2 = 0;
  axis normal
  axis off
  set(get(fig,'Children'),'XLim',[-1-width 1+width]);
  set(get(fig,'Children'),'YLim',[0 1]);
  drawnow

else
  if package(1)==1
    % PLAYER 1
    if package(3)<0, package(3)=256+package(3);end
    if package(3)-no1>1
      fprintf('PLAYER 1 looses %i packages\n',package(3)-no1-1);
    end
    no1 = package(3);
    fb_opt.racket1_x= [-width;width]+package(2);
    
    set(fb_opt.h_racket1, 'xData',fb_opt.racket1_x);
  else
    %PLAYER 2
    if package(3)<0, package(3)=256+package(3);end
    if package(3)-no2>1
      fprintf('PLAYER 2 looses %i packages\n',package(3)-no2-1);
    end
    no2 = package(3);
    fb_opt.racket2_x= [-width;width]+package(2);
    
    set(fb_opt.h_racket2, 'xData',fb_opt.racket2_x);
  end
  set(get(fig,'Children'),'XLim',[-1-width 1+width]);
  set(get(fig,'Children'),'YLim',[0 1]);
  drawnow;
end
