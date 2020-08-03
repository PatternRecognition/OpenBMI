function play_brainpong(port);

if ~exist('port','var')
  port = 12423;
end

[aaa,sss] = system('hostname');

old_pos1 = 0;
old_pos2 = 0;
ori = 0;

START = 10;
STOP = 11;
PAUSE = 12;
EXIT = 13;
PLAYER1 = 1;
PLAYER2 = 2;
BALL = 5;
SCORE_ON = 7;
SCORE1 = 8;
SCORE2 = 9;
BALL_ON = 20;
DIAM = 21;
BALL_COLOR = 22;
BACKGROUND_COLOR = 27;
BAT_COLOR1 = 31;
BAT_COLOR2 = 41;
BAT_WIDTH1 = 32;
BAT_WIDTH2 = 42;
BAT_HEIGHT1 = 33;
BAT_HEIGHT2 = 43;
COUNTER_ON = 50;
COUNTER = 51;
POSITION = 70;
VIEW = 60;
WINNER = 77;

run = 1;

batwidth = 1;
batheight = 0.1;

batwidth1 = batwidth;
batwidth2 = batwidth;
batheight1 = batheight;
batheight2 = batheight;

clf;
scs = get(0,'ScreenSize')*0.5;
set(gcf,'Position',scs);
set(gcf,'Menubar','none');
set(gcf,'DoubleBuffer','on');
set(gcf,'NumberTitle','off');
%set(gcf,'Units','normalized');
set(gcf,'Color',[0.9,0.9,0.9]);
set(gca,'XLim',[-1 1]);
set(gca,'YLim',[-1 1]);
set(gca,'Position',[0 0 1 1]);
axis off;
bat1 = patch([-0.5*batwidth,-0.5*batwidth,0.5*batwidth,0.5*batwidth],[-1,-1+batheight,-1+batheight,-1],[0.5,0.5,0.5]);
set(bat1,'EdgeColor','none','EraseMode','xor')
bat2 = patch([-0.5*batwidth,-0.5*batwidth,0.5*batwidth,0.5*batwidth],[1,1-batheight,1-batheight,1],[0.5,0.5,0.5]);
set(bat2,'EdgeColor','none','EraseMode','xor')
count = text(0,0,'');
set(count,'FontUnits','normalized','FontSize',0.2,'HorizontalAlignment','center','VerticalAlignment','middle','Color',[0 0 0],'Visible','off');
if ori 
    set(count,'Rotation',90);
end 
    
score1 = text(-0.95,1-batheight2,'');
set(score1,'FontUnits','normalized','FontSize',0.1,'HorizontalAlignment','left','VerticalAlignment','top','Color',[0 0 0],'Visible','off');

score2 = text(0.95,1-batheight2,'');
set(score2,'FontUnits','normalized','FontSize',0.1,'HorizontalAlignment','right','VerticalAlignment','top','Color',[0 0 0],'Visible','off');

if ori
    set(score1,'Rotation',90);
    set(score2,'Rotation',90);
end

ball_x = 0;
ball_y = 0;

circle_x = linspace(0,2*pi,32);
circle_y = sin(circle_x);
circle_x = cos(circle_x);
if ori==1
    circle_y = circle_y/scs(3)*scs(4);
else
    circle_x = circle_x/scs(3)*scs(4);
end
circ_x = 0.025*circle_x;
circ_y = 0.025*circle_y;
    
ball = patch(circle_x*0.05,circle_y*0.05,[0.5,0.5,0.5],'EraseMode','xor');
set(ball,'EdgeColor','none');
dia = 0.05;

drawnow;

get_udp(sss(1:end-1),{3;1;4;4;4;4},port,2);

blold = [];
while run
  a = get_udp(0);
  while isempty(a)    
    drawnow;
    a = get_udp(0);
  end 
  if ~isempty(blold) & a(1)>blold+1
    fprintf('Loss %d packages\n',a(1)-blold-1);
  end
  blold = a(1);
  switch a(2)
   case POSITION
    set(gcf,'Position',a(3:end));
    scs = a(3:end);
   case PLAYER1
    set(bat1,'XData',a(3)*(1-0.5*batwidth1)+[-0.5*batwidth1,-0.5*batwidth1,0.5*batwidth1,0.5*batwidth1]);
    old_pos1 = a(3);
    
   case PLAYER2
    set(bat2,'XData',-a(3)*(1-0.5*batwidth2)+[-0.5*batwidth2,-0.5*batwidth2,0.5*batwidth2,0.5*batwidth2]);
    old_pos2 = a(3);
    
   case VIEW 
    ori = a(3);
    circle_x = linspace(0,2*pi,32);
    circle_y = sin(circle_x);
    circle_x = cos(circle_x);
    if ori
        circle_y = circle_y/scs(3)*scs(4);
        set(gca,'View',[-90 90]);
        set(score1,'Rotation',90);
        set(score2,'Rotation',90);
        set(count,'Rotation',90);
    else
        circle_x = circle_x/scs(3)*scs(4);
        set(gca,'View',[0 90]);
        set(score1,'Rotation',0);
        set(score2,'Rotation',0);
        set(count,'Rotation',0);
    end    
    circ_x = 0.5*circle_x*dia;
    circ_y = 0.5*circle_y*dia;
    
case START 
    %nothing???
    
   case STOP
    set(count,'Visible','on','String','stopped');
    
   case PAUSE
    set(count,'Visible','on','String','paused');
    
   case BACKGROUND_COLOR
    set(gcf,'Color',a(3:end-1));
    
   case BAT_COLOR1
    set(bat1,'FaceColor',a(3:end-1));
    
   case BAT_COLOR2
    set(bat2,'FaceColor',a(3:end-1));
    
   case BAT_WIDTH1 
    batwidth1 = a(3);
    set(bat1,'XData',old_pos1*(1-0.5*batwidth1)+[-0.5*batwidth1,-0.5*batwidth1,0.5*batwidth1,0.5*batwidth1]);
    
    
   case BAT_WIDTH2 
    batwidth2 = a(3);
    set(bat2,'XData',-old_pos2*(1-0.5*batwidth2)+[-0.5*batwidth2,-0.5*batwidth2,0.5*batwidth2,0.5*batwidth2]);
        
    
   case BAT_HEIGHT1
    set(bat1,'YData',[-1,-1+a(3),-1+a(3),-1]);
    batheight1 = a(3);
    
   case BAT_HEIGHT2
    set(bat2,'YData',[1,1-a(3),1-a(3),1]);
    bla1 = get(score1,'Position');
    bla2 = get(score2,'Position');
    bla1(2) = 1-a(3);
    bla2(2) = 1-a(3);
    set(score1,'Position',bla1);
    set(score2,'Position',bla2);
    batheight2 = a(3);
    
case COUNTER_ON 
    if a(3)
      set(count,'Visible','on','String','');
    else
      set(count,'Visible','off');
    end
    
   case COUNTER 
    set(count,'String',int2str(a(3)));
    
   case BALL
    ball_x = a(3);
    ball_y = a(4);
    try
        set(ball,'XData',circ_x+ball_x);
        set(ball,'YData',circ_y+ball_y);
    end
    
   case BALL_ON
    if a(3)
      set(ball,'Visible','on');
    else
      set(ball,'Visible','off');
    end
    
   case DIAM 
    circ_x = 0.5*circle_x*a(3);
    circ_y = 0.5*circle_y*a(3);
    dia = a(3);
    set(ball,'XData',circ_x+ball_x);
    set(ball,'YData',circ_y+ball_y);
    
    
   case BALL_COLOR
    set(ball,'FaceColor',a(3:end-1));
    
    
   case SCORE_ON
    if a(3)
        set(score1,'Visible','on');
        set(score2,'Visible','on');
    else
        set(score1,'Visible','off');
        set(score2,'Visible','off');
    end
    
   case SCORE1
    set(score1,'String',sprintf('% 2d',a(3)));
    
   case SCORE2
    set(score2,'String',sprintf('% 2d',a(3)));

   case WINNER
    if a(3)
        set(count,'String','Winner','Visible','on');
    else
        set(count,'String','Loser','Visible','on');
    end
    
   case EXIT
    run = false;
    
    
  end
  

end

get_udp('close');
close all


