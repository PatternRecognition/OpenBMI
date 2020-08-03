function fb_opt = feedback_brainpong_master(fb_opt, dat1,dat2);
% 31.1.2006 by Guido
% called by matlab_feedbacks_master
% sets changes in graphic objects, calls clients for visualization

% handle number in feedback_brainpong_client
ball = 1;
button1 = 2;
button2 = 3;
countdown = 4;
win1 = 9;
win2 = 10;
axis = 11;
fig = 12;

global DATA_DIR

persistent x_coor y_coor bat1 bat2 status time x_ball y_ball ball_pos bat_change score1 score2 looser winkel score old_status lin played

fb_opt = set_defaults(fb_opt,'reset',1);

if fb_opt.reset==1
  score1 = [5,7];
  score2 = [6,8];
  lin = [13,14];
  fb_opt = set_defaults(fb_opt,'countdown',3000,...
    'radius',0.1,...
    'portrait',1,...
    'background_color',[0 0 0],...
    'position',[1600,0,1280,1024],...
    'ball_color',[0.8 0.8 0.8],...
    'ball_free_time',1000,...
    'ball_miss_free_time',1000,...
    'reflexion_mode',0,...
    'sideways',0,...
    'ball_modus',0,...
    'ball_miss_color',[1 0 0],...
    'matchpoints',inf,...
    'middleline',0,...                           
    'speed',0.4,...
    'sound',false,...
    'border_sound',[DATA_DIR 'sound/E5.wav'],...
    'ballout_sound', [DATA_DIR 'sound/A5.wav'],...
    'ball',1,...                           
    'noise',0,...
    'parPort',1,...
    'status','pause',...
    'player1',struct,...
    'player2',struct,...
    'changed',0,...
    'init_file','feedback_brainpong_client',...
    'log',1,...
    'show_score',1);
  
  fb_opt.player1 = set_defaults(fb_opt.player1,...
    'bat_width',0.5,...
    'bat_color',[0 1 0],...
    'bat_sound', [DATA_DIR 'sound/G5.wav'],...
    'relational',0,...
    'damping',20,...
    'bat_height',0.1);
  
  fb_opt.player2 = set_defaults(fb_opt.player2,...
    'bat_width',0.5,...
    'bat_color',[0 0 1],...
    'bat_sound',[DATA_DIR 'sound/G5.wav'],...
    'relational',0,...
    'damping',20,...
    'bat_height',0.1);
  
  % each client could have differnet window positions
  if size(fb_opt.position,1)==1
    fb_opt.position = repmat(fb_opt.position,[length(fb_opt.client_machines),1]);
  end

  % load sounds
  [a,b] = wavread(fb_opt.border_sound);
  fb_opt.border_sound_datei = struct('name',fb_opt.border_sound,'data',a,'fs',b);
  [a,b] = wavread(fb_opt.ballout_sound);
  fb_opt.ballout_sound_datei = struct('name',fb_opt.ballout_sound,'data',a,'fs',b);
  [a,b] = wavread(fb_opt.player1.bat_sound);
  fb_opt.player1.bat_sound_datei = struct('name',fb_opt.player1.bat_sound,'data',a,'fs',b);
  [a,b] = wavread(fb_opt.player2.bat_sound);
  fb_opt.player2.bat_sound_datei = struct('name',fb_opt.player2.bat_sound,'data',a,'fs',b);
  
  % check resolution of screen
  rela = fb_opt.position(:,3)./fb_opt.position(:,4);
  if ~all(rela==rela(1)),
    warning('please use the same side proportions for all clients');
  end
  rela = rela(1);
  
  
  % initialize do_set
  handle = [ball,button1,button2,countdown,score1,score2,win1,win2,axis];
  % client receives much information about which player number it becomes, opens ports, etc.
  % last argument: name of function to call as client during processing data
  % "brainpong_master" becomes part of the log file name
  do_set('init',handle,'brainpong_master',fb_opt,'feedback_brainpong_client');
  
  % get round shape for ball
  if fb_opt.ball
    x_coor = fb_opt.radius*cos(2*pi/60*(1:60));
    y_coor = rela*fb_opt.radius*sin(2*pi/60*(1:60));
  else
    x_coor = [-1 1 1 -1]*fb_opt.radius;
    y_coor = [-1 -1 1 1]*rela*fb_opt.radius;
  end
  x_ball = fb_opt.radius;
  y_ball = rela*x_ball;
  
  % define bat positions
  if fb_opt.portrait
    bat1 = cat(1,[-0.5 0.5 0.5 -0.5]*fb_opt.player1.bat_width, [0 0 1 1]*fb_opt.player1.bat_height-1);
    bat2 = cat(1,[-0.5 0.5 0.5 -0.5]*fb_opt.player2.bat_width, [-1 -1 0 0]*fb_opt.player2.bat_height+1);
    bat_change = 'XData';  
  else
    bat1 = cat(1,[-1 -1 0 0]*fb_opt.player1.bat_height+1,[-0.5 0.5 0.5 -0.5]*fb_opt.player1.bat_width);
    bat2 = cat(1,[0 0 1 1]*fb_opt.player2.bat_height-1,[-0.5 0.5 0.5 -0.5]*fb_opt.player2.bat_width);
    bat_change = 'YData';
  end
  
  ball_pos = [0,0];
  % initialize figures
  do_set(ball,'Visible','off','FaceColor',fb_opt.ball_color,'Xdata',ball_pos(1)+x_coor,'YData',ball_pos(2)+y_coor);
  do_set(countdown,'Visible','on','Color',fb_opt.ball_color,'String',int2str(ceil(fb_opt.countdown/1000)));
  do_set(score1,'String','0','Color',fb_opt.player1.bat_color);
  do_set(score2,'String','0','Color',fb_opt.player2.bat_color);
  do_set(score1,'Visible','off');
  do_set(score2,'Visible','off');
  do_set(win1,'Visible','off');
  do_set(win2,'Visible','off');

  % set bat positions
  do_set([button1],'Visible','on','XData',bat1(1,:),'YData',bat1(2,:),'FaceColor',fb_opt.player1.bat_color);
  do_set([button2],'Visible','on','XData',bat2(1,:),'YData',bat2(2,:),'FaceColor',fb_opt.player2.bat_color);
  
  if ~fb_opt.portrait
    bat1 = bat1([2,1],:);
    bat2 = bat2([2,1],:);
    if ~fb_opt.sideways
      do_set(countdown,'Rotation',90);
      do_set(win1,'Rotation',90);
      do_set(win2,'Rotation',90);
    end
  end
  
  if fb_opt.portrait 
    do_set(score1(2),'Visible','off');
    do_set(score2(2),'Visible','off');
    do_set(lin(2),'Visible','off');
    score1 = score1(1);
    score2 = score2(1);
    lin = lin(1);
  else
    do_set(score1(1+fb_opt.sideways),'Visible','off');
    do_set(score2(1+fb_opt.sideways),'Visible','off');
    do_set(lin(1),'Visible','off');
    score1 = score1(2-fb_opt.sideways);
    score2 = score2(2-fb_opt.sideways);
    if fb_opt.sideways
      bla = score1;
      score1 = score2;
      score2 = bla;
    end
    lin = lin(2);
  end
  
  if fb_opt.middleline
    do_set(lin,'Visible','on');
  else
    do_set(lin,'Visible','on');
  end
  
  do_set(fig,'Color',fb_opt.background_color);
    % other initializiations

  % status defines intern state of the function
  %  0 : countdown 
  %  1 : wait for freeing the ball
  %  2 : play
  %  3 : miss ->1
  %  4 : after matchpoints
  status = 0;
  played = 0;
  score = [0,0];
  % time is everything
  tic;
  time = toc;
  old_status = 'pause';
  fb_opt.status = 'pause';
 
end

ti = toc;

fb_opt.reset = 0;

if fb_opt.changed 
  % which options shall be settable via a simple "send"? (Ball speed?)
  % catch game mode, e.g. pause induced by supervisor

  if status==3
    do_set(ball,'FaceColor',fb_opt.ball_miss_color);
  else
    do_set(ball,'FaceColor',fb_opt.ball_color);
  end

  if fb_opt.middleline
    do_set(lin,'Visible','on');
  else
    do_set(lin,'Visible','on');
  end

  if fb_opt.show_score & status>=1
    do_set([score1,score2],'Visible','on');
  end
  
  if ~fb_opt.show_score
    do_set([score1,score2],'Visible','off');
  end
  
  if fb_opt.portrait
    bat1 = cat(1,[-0.5 0.5 0.5 -0.5]*fb_opt.player1.bat_width, [0 0 1 1]*fb_opt.player1.bat_height-1);
    bat2 = cat(1,[-0.5 0.5 0.5 -0.5]*fb_opt.player2.bat_width, [-1 -1 0 0]*fb_opt.player2.bat_height+1);
    bat_change = 'XData';  
  else
    bat1 = cat(1,[-1 -1 0 0]*fb_opt.player1.bat_height+1,[-0.5 0.5 0.5 -0.5]*fb_opt.player1.bat_width);
    bat2 = cat(1,[0 0 1 1]*fb_opt.player2.bat_height-1,[-0.5 0.5 0.5 -0.5]*fb_opt.player2.bat_width);
    bat_change = 'YData';
  end
  do_set([button1],'XData',bat1(1,:),'YData',bat1(2,:),'FaceColor',fb_opt.player1.bat_color);
  do_set([button2],'XData',bat2(1,:),'YData',bat2(2,:),'FaceColor',fb_opt.player2.bat_color);
  if ~fb_opt.portrait
    bat1 = bat1([2,1],:);
    bat2 = bat2([2,1],:);
  end
  
  
  if ~strcmp(old_status,fb_opt.status)
    switch fb_opt.status
     case 'pause'
      status = 4;
      do_set([score1,score2,countdown,win1,win2,ball],'Visible','off');
     case 'play'
      if played
        status = 1;
        do_set(countdown,'Visible','off');
        time = ti;
        do_set(ball,'Visible','on');
        if fb_opt.show_score      
          do_set(score1,'Visible','on');
          do_set(score2,'Visible','on');
        end
        if isempty(looser)
          looser = 1+(randn>0); % which player starts?
        end
      else
        status = 0;
        time = ti;
      end
    end
  end
  
  old_status = fb_opt.status;
end

fb_opt.changed = 0;


% Calculate new bat positions
if ~isempty(dat1) % bat 1
  if fb_opt.player1.relational
    rel = dat1/fb_opt.player1.damping;
  else
    rel = dat1*(1-fb_opt.player1.relational)/(1-fb_opt.player1.relational^fb_opt.player1.damping);
  end
  bat1(1,:) = bat1(1,:)-mean(bat1(1,:))+fb_opt.player1.relational*mean(bat1(1,:))+rel;
  
  if max(bat1(1,:))>1
    bat1(1,:) = bat1(1,:)-max(bat1(1,:))+1;
  end
  if min(bat1(1,:))<-1
    bat1(1,:) = bat1(1,:)-min(bat1(1,:))-1;
  end
end

if ~isempty(dat2) % bat 2
  if fb_opt.player2.relational
    rel = dat2/fb_opt.player2.damping;
  else
    rel = dat2*(1-fb_opt.player2.relational)/(1-fb_opt.player2.relational^fb_opt.player2.damping);
  end
  bat2(1,:) = bat2(1,:)-mean(bat2(1,:))+fb_opt.player2.relational*mean(bat2(1,:))+rel;
  
  if max(bat2(1,:))>1
    bat2(1,:) = bat2(1,:)-max(bat2(1,:))+1;
  end
  if min(bat2(1,:))<-1
    bat2(1,:) = bat2(1,:)-min(bat2(1,:))-1;
  end
end

% prepare values for bat posisions, setting is done later (see do_set('+'))
do_set(button1,bat_change,bat1(1,:));
do_set(button2,bat_change,-bat2(1,:));


if ~strcmp(fb_opt.status,'play')
  do_set('+');
  return; % if pause is on, only bat positions are set, nothing else...
end

% deal with countdown at beginning of game
if status==0
  if ti-time<fb_opt.countdown/1000
    do_set(countdown,'String',ceil(fb_opt.countdown/1000-ti+time));
  else
    do_set(countdown,'Visible','off');
    time = ti;
    do_set(ball,'Visible','on');
    status = 1;
    if fb_opt.show_score      
      do_set(score1,'Visible','on');
      do_set(score2,'Visible','on');
    end
    looser = 1+(randn>0); % which player starts?
  end
end


if status == 1
  played = 1;
  % place ball at one bat, wait for start
    switch looser
      case 1 % player that has the ball
        if fb_opt.portrait
          ball_pos(2) = -1+(y_ball+fb_opt.player1.bat_height);
        else
          ball_pos(1) = 1-x_ball-fb_opt.player1.bat_height;
        end
      case 2
        if fb_opt.portrait
          ball_pos(2) = 1-(y_ball+fb_opt.player2.bat_height);
        else
          ball_pos(1) = -1+(x_ball+fb_opt.player1.bat_height);
        end
    end

    switch fb_opt.ball_modus 
      case 0 % ball is on the bat
       if looser==1
          dum = mean(bat1(1,:));
        else
          dum = -mean(bat2(1,:));
        end
        ball_pos(2-fb_opt.portrait) = dum;
        winkel = (rand*0.6+0.2)*pi;
        if ~fb_opt.portrait
          winkel = winkel-pi/2;
        end
        if looser==2
          winkel = -winkel;
        end
        winkel = [cos(winkel),sin(winkel)];
     case 1 % ball is in one edge/winkel is nice
      if looser>0
        if fb_opt.portrait
          dum =  0.5*(x_ball+1);
          winkel = [2*(dum+1-2*x_ball),2-fb_opt.player1.bat_height-fb_opt.player2.bat_height-2*y_ball];
        else
          dum =  0.5*(y_ball+1);
          winkel = [2-fb_opt.player1.bat_height-fb_opt.player2.bat_height-2*x_ball,2*(dum+1-2*y_ball)];
        end
        winkel = winkel+fb_opt.noise*randn(1,2);
        winkel = winkel/sqrt(sum(winkel.^2));
        winkel(fb_opt.portrait+1) = winkel(fb_opt.portrait+1)*sign(1.5-looser)*(2*fb_opt.portrait-1);
        winkel(2-fb_opt.portrait) = sign(randn)*winkel(2-fb_opt.portrait);
        if randn>0
          dum = dum-1;
        else
          dum = 1-dum;
        end
        ball_pos(2-fb_opt.portrait) = dum;
        looser = 0;
      end
     otherwise
        error('implement more ball start strategies');
    end
    
    if ti-time>=fb_opt.ball_free_time/1000
      status = 2;
      time = ti;
    end
end

if status == 2
  % playing around
  % looking for border events
  
  side = 1; side2 = 1;
  while ~isempty(winkel) & (~isempty(side) | ~isempty(side2))
    bew = ti-time;
    ball_old = ball_pos;
    ball_pos = ball_pos+bew*winkel*fb_opt.speed;
    % left and right border
    if fb_opt.portrait
      if abs(ball_pos(1))+x_ball>1
        side = abs((sign(ball_pos(1))*(1-x_ball)-ball_old(1))/winkel(1)/fb_opt.speed);
      else
        side = [];
      end
    else
      if abs(ball_pos(2))+y_ball>1
        side = abs((sign(ball_pos(2))*(1-y_ball)-ball_old(2))/winkel(2)/fb_opt.speed);
      else
        side = [];
      end
        
    end
    
    % upper and lower border
    if fb_opt.portrait
      if ball_pos(2)+y_ball+fb_opt.player2.bat_height>1 & winkel(2)>0
        side2 = -abs((1-y_ball-fb_opt.player2.bat_height-ball_old(2))/winkel(2)/fb_opt.speed);
      elseif ball_pos(2) -y_ball-fb_opt.player1.bat_height<-1 & winkel(2)<0
        side2 = abs((-1+y_ball+fb_opt.player1.bat_height-ball_old(2))/winkel(2)/fb_opt.speed);
      else
        side2 = [];
      end
    else
      if ball_pos(1)+x_ball+fb_opt.player2.bat_height>1  & winkel(1)>0
        side2 = abs((1-x_ball-fb_opt.player2.bat_height-ball_old(1))/winkel(1)/fb_opt.speed);
      elseif ball_pos(1) -x_ball-fb_opt.player1.bat_height<-1 & winkel(1)<0
        side2 = -abs((-1+x_ball+fb_opt.player1.bat_height-ball_old(1))/winkel(1)/fb_opt.speed);
      else
        side2 = [];
      end
    end
    
    if ~isempty(side) & ~isempty(side2)
      if side<abs(side2) 
        side2 = [];
      else
        side = [];
      end
    end
    
    if ~isempty(side)
      ball_pos = ball_old+side*winkel*fb_opt.speed;
      time = side+time;
      if fb_opt.portrait
        winkel(1) = -winkel(1);
      else
        winkel(2) = -winkel(2);
      end
      if fb_opt.sound
        do_set('sound',fb_opt.border_sound_datei.data,fb_opt.border_sound_datei.fs,fb_opt.border_sound_datei.name);
%wavplay(fb_opt.border_sound_datei.data,fb_opt.border_sound_datei.fs,'async');
end
    end
    
    if ~isempty(side2)
      ball_pos = ball_old+abs(side2)*winkel*fb_opt.speed;
      time = abs(side2)+time;
      if fb_opt.portrait
        rel_pos = ball_pos(1);
        rel_win = winkel(2);
        rel_ball = x_ball;
      else
        rel_pos = ball_pos(2);
        rel_win = winkel(1);
        rel_ball = y_ball;
      end
      if side2<0
        % check player2
        ww = check_hit(rel_pos,-bat2(1,:),rel_win,rel_ball,fb_opt);
      else
        ww = check_hit(rel_pos,bat1(1,:),rel_win,rel_ball,fb_opt);
      end
      
      if isempty(ww)
        status = 3;
        winkel = [];
        do_set(ball,'FaceColor',fb_opt.ball_miss_color);
        if fb_opt.sound
          do_set('sound',fb_opt.ballout_sound_datei.data,fb_opt.ballout_sound_datei.fs,fb_opt.ballout_sound_datei.name);
        end
        if side2<0
          looser = 2;
          score(1) = score(1)+1;
          do_set(score1,'String',sprintf('%i',score(1)));
        else
          looser = 1;
          score(2) = score(2)+1;
          do_set(score2,'String',sprintf('%i',score(2)));
        end
        if max(score)>=fb_opt.matchpoints
          status = 4;
          if score(1)>score(2)
            do_set(win1,'Visible','on','String','You win')
            do_set(win2,'Visible','on','String','You loose');
          elseif score(1)<score(2)
            do_set(win2,'Visible','on','String','You win')
            do_set(win1,'Visible','on','String','You loose');
          end
        end

      else
        if fb_opt.portrait
          winkel(2) = ww;
        else
          winkel(1) = ww;
        end
        if fb_opt.sound
          if side2>0
            do_set('sound',fb_opt.player1.bat_sound_datei.data,fb_opt.player1.bat_sound_datei.fs,fb_opt.player1.bat_sound_datei.name);
          else
            do_set('sound',fb_opt.player2.bat_sound_datei.data,fb_opt.player1.bat_sound_datei.fs,fb_opt.player1.bat_sound_datei.name);
          end
        end
      end
    end
    
    
  end
  time = ti;
end

if status == 3
  % miss a ball or stop, prepare for status = 1
  if ti - time>=fb_opt.ball_miss_free_time/1000
    do_set(ball,'FaceColor',fb_opt.ball_color);
    time = ti;
    status = 1;
  end
end

if status==4
  % nothing to do
end

do_set(ball,'XData',x_coor+ball_pos(1),'YData',y_coor+ball_pos(2));

do_set('+');

return;




function ww = check_hit(ball,bat,winkel,rad,fb_opt);
%TODO: extend by interesting reflexion_modes

switch fb_opt.reflexion_mode
  case 0
    bat = [min(bat)-rad,max(bat)+rad];
    if ball>=bat(1) & ball<=bat(2)
      ww = -winkel;
    else
      ww = [];
    end
    
  otherwise
    error('no other reflexion modes available')
end

return;
