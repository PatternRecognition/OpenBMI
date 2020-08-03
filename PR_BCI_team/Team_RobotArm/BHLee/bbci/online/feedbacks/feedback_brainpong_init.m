function handle = feedback_brainpong_init(fig,fb_opt);

fast_axis= {'clipping','off', 'hitTest','off', 'interruptible','off', ...
    'drawMode','fast'};
fast_obj= {'EraseMode','normal','hitTest','off', 'interruptible','off'};

if isunix 
  const = 2.8;
else
  const = 2.25;
end

clf;
set(fig, 'Menubar','none', 'Resize','off', ...
         'position',fb_opt.position);
hold on;
set(fig,'DoubleBuffer','on', 'BackingStore','off',...
        'Renderer','OpenGL','RendererMode','auto',...
        'Pointer','custom', 'PointerShapeCData',ones(16)*NaN);
set(gca, 'position',[0 0 1 1]);
set(gca, fast_axis{:});
axis off
set(fig,'Color',fb_opt.background_color);


fac = fb_opt.bat_width/(1-fb_opt.bat_width);
fac = [-fac,fac,fac,-fac];

bat = patch(fac,[0 0 fb_opt.bat_height fb_opt.bat_height],'r');
set(bat,'FaceColor',fb_opt.bat_color, 'EdgeColor','none');
ball = plot(0,1-0.5*fb_opt.ball_diameter,'.');
set(ball,'MarkerSize',fb_opt.position(4)*fb_opt.ball_diameter*const);
set(ball,'MarkerEdgeColor',fb_opt.ball_color,fast_obj{:});
set(ball,'Visible','off');
set(gca, 'XLim',[-1/(1-fb_opt.bat_width),1/(1-fb_opt.bat_width)]);
set(gca, 'YLim',[0,1]);

ht= text(-0.5,0.99,'HIT: 0');
ht(2)= text(0.5,0.99,'MISS: 0');
set(ht,'VerticalAlignment','top', 'HorizontalAlignment','center', ...
       'FontSize',40);

if ~fb_opt.score
  set(ht,'Visible','off');
end

axis off

cou = text(0, 0.5, int2str(ceil(fb_opt.countdown/1000)));
set(cou,'HorizontalAlignment','center', 'FontSize',100);

handle = [bat,ball,cou,ht,fig,gca];

