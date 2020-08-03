function fb_opt = feedback_brainpong(fig, fb_opt, player, x);

persistent fbb_old bat fac ball status counter cou chbadi xpos seq ...
    hit miss xold xposi bit ht ballpos chbahi diam bahi gaMo verz

if isempty(fb_opt),
  fb_opt= struct('reset',1);
end

neustart = 0;

if fb_opt.reset,
fb_opt = set_defaults(fb_opt, ...
                      'status','pause',...
                      'bat_width',1/3,...
                      'bat_height',1/10,...
                      'ball_diameter',1/10,...
                      'speed',500,...
                      'ball_color',[0 0 1],...
                      'ball_hit_color',[0 1 0],...
                      'ball_miss_color',[1 0 0],...
                      'bat_color',[0.6,0.6,0.6],...
                      'ball_positions',2,...
                      'changed',0,...
                      'show_bit',0,...
                      'score',1,...
                      'damping',20,...
                      'countdown',5000,...
                      'fs',25,...
                      'reset',1,...
                      'sequence','Z',...
                      'relational',0,...
                      'parPort',1,...
                      'log',0,...
                      'position',[100 100 800 600],...
                      'background_color',[0.9,0.9,0.9],...
                      'matchpoints',100,...
                      'repetitions',1,...
                      'interbreak_interval',1000);
if length(fb_opt.ball_positions)==1 & ...
      round(fb_opt.ball_positions)==fb_opt.ball_positions & ...
   fb_opt.ball_positions>=2
  fb_opt.ball_positions = linspace(-1, 1, fb_opt.ball_positions);
end
  fac = fb_opt.bat_width/(1-fb_opt.bat_width);
  fac = [-fac,fac,fac,-fac]

  verz = 2/(1-fb_opt.bat_width)*fb_opt.position(4)/fb_opt.position(3);

  hit = 0;
  miss = 0;

  fbb_old = fb_opt;
  fbb_old.status = '';
  fb_opt.reset = 0;
  status = 0;
  xold = 0;

  ballpos = [NaN,NaN];
  diam = fb_opt.ball_diameter;
  bahi = fb_opt.bat_height;
  neustart=1;

  handle = feedback_brainpong_init(fig,fb_opt);
  bat = 1;ball = 2; cou = 3;ht = [4,5]; fig=6; gc = 7;
  
  do_set('init',handle,'brainpong',fb_opt);
  do_set(200);

end
 
if fb_opt.changed==1 
  if fbb_old.bat_width~=fb_opt.bat_width
    fac = fb_opt.bat_width/(1-fb_opt.bat_width);
    fac = [-fac,fac,fac,-fac];
    do_set(gc,'XLim',[-1/(1-fb_opt.bat_width),1/(1-fb_opt.bat_width)]);
    do_set(bat,'XData',fac);
    verz = 2/(1-fb_opt.bat_width)*fb_opt.position(4)/fb_opt.position(3);
  end

  if any(fbb_old.background_color~=fb_opt.background_color)
    do_set(fig,'Color',fb_opt.background_color);
  end

  if fbb_old.bat_height~=fb_opt.bat_height
    chbahi = 1;
  end

  if any(fbb_old.bat_color~=fb_opt.bat_color)
    do_set(bat,'FaceColor',fb_opt.bat_color);
  end
  
  if fbb_old.ball_diameter~=fb_opt.ball_diameter
    chbadi = 1;
  end
  
  if fb_opt.score
    do_set(ht(1),'Visible','on');
    do_set(ht(2),'Visible','on');
  else
    do_set(ht(1),'Visible','off');
    do_set(ht(2),'Visible','off');
  end
end

if fb_opt.changed | neustart 
  if ~strcmp(fbb_old.status,fb_opt.status)
    switch fb_opt.status
     case 'play'
      if status>=10
        status = status-10;
      end        
      if status==0
        status = 1;
        hit = 0;miss = 0;
        counter = round(fb_opt.countdown*fb_opt.fs/1000);
        do_set(cou,'String',int2str(ceil(fb_opt.countdown/fb_opt.fs)),'Visible','on');
      end
      do_set(210);
     case 'pause'
      status = 10+status;
      if status==0
        hit = 0;miss = 0;
      end
      do_set(cou,'String','pause', 'Visible','on');
      do_set(211);
     case 'stop'
      status = 0;
      do_set(ball,'Visible','off');
      if ~isempty(bit) & fb_opt.ball_positions>=2
        do_set(cou,'String',sprintf('%2.1f bit/min',bit),'Visible','on');
      else
        do_set(cou,'String','stopped', 'Visible','on');
      end
      do_set(212);
    end
  end
end

fbb_old = fb_opt;

fb_opt.changed = 0;



if status == 0
  for i = 1:fb_opt.repetitions
    do_set('+');
  end
  return
end


if fb_opt.relational==1
  xold = fb_opt.relational*xold+x/fb_opt.damping;
else
  xold = fb_opt.relational*xold+x*(1-fb_opt.relational)/(1-fb_opt.relational^fb_opt.damping);
end

  
if fb_opt.relational>0
    if status==3 | status==13
      if fb_opt.relational==1
        xold = xold+x/fb_opt.damping;
      else
        xold = fb_opt.relational*xold+x*(1-fb_opt.relational)/(1-fb_opt.relational^fb_opt.damping);
      end
      xold = max(min(xold,1),-1);  
    elseif status>=10 | status==4
        % stay at the old position
    else 
        xold = 0;
    end   
else
    xold = x;
end

do_set(bat,'XData',fac+xold);

if status>=10
  for i = 1:fb_opt.repetitions;
    do_set('+');
  end
  return
end

numb = 0;

if status == 1
  % countdown
  if counter<=0
    do_set(cou,'Visible','off');
    status = 2;
    do_set(ball,'Visible','on');
    seq = length(fb_opt.sequence)-1;
    do_set(201);
    tic;
end
  counter = counter-1;
  do_set(cou,'String',int2str(ceil(counter/fb_opt.fs)));
end


if status == 2
  % ball position
  if chbadi
    do_set(ball,'MarkerSize',fb_opt.position(4)*fb_opt.ball_diameter*const);
    chbadi = 0;
    diam = fb_opt.ball_diameter;
  end
  if chbahi
    do_set(bat,'YData',[0 0 fb_opt.bat_height fb_opt.bat_height]);
    chbahi = 0;
    bahi = fb_opt.bat_height;
  end
  
  do_set(ball,'MarkerEdgeColor',fb_opt.ball_color);
  status = 3;numb = 0;
  if length(fb_opt.ball_positions)>1
    counter = -(1-diam-bahi)/ceil(fb_opt.speed*fb_opt.fs/1000);
    seq = mod(seq+1,length(fb_opt.sequence))+1;
    dec = fb_opt.sequence(seq);
    if strcmp(lower(dec),'z')
      xposi = ceil(length(fb_opt.ball_positions)*rand);    
    else    
      xposi = dec-'0';
    end
    ballpos = [fb_opt.ball_positions(xposi),1+0.5*diam;0,counter];
    gaMo = 0;
  else
    % freies Spiel
    % erster Schuss nach links oder rechts????
    gaMo = 1;
    di = sign(randn(1));
    xposi = 0;
    if fb_opt.ball_positions<1
      ballpos(1,1)= di*(fb_opt.bat_width/(1-fb_opt.bat_width)-0.5*diam*verz);
      ballpos(1,2)= 1-0.5*diam;
      ballpos(2,1)=di*(-1+0.5*diam*verz-fb_opt.bat_width/(1-fb_opt.bat_width))+fb_opt.ball_positions*randn;
      ballpos(2,2)=-(1-diam-bahi);
      ballpos(2,:) = ballpos(2,:)/ceil(fb_opt.speed*fb_opt.fs/1000);
    else
      ballpos(1,1)= di;
      ballpos(1,2) = 0.5*diam+bahi;
      ballpos(2,1)=di*(-1+0.5*diam*verz-fb_opt.bat_width/(1-fb_opt.bat_width))+(fb_opt.ball_positions-1)*randn/2;
      ballpos(2,2)=(1-diam-bahi);
      ballpos(2,:) = ballpos(2,:)/ceil(fb_opt.speed*fb_opt.fs/1000);
    end
    ballpos(1,:) = ballpos(1,:)-ballpos(2,:)/fb_opt.repetitions;
  end
  do_set(max(1,xposi));
end

if status == 3
  % ball falls
  for j = 1:fb_opt.repetitions
    ballpos(1,:)= ballpos(1,:)+ballpos(2,:)/fb_opt.repetitions;
    % Ball am linken Rand
    if ballpos(1,1)<=-1/(1-fb_opt.bat_width)+0.5*diam*verz & ballpos(2,1)<0
      ballpos(1,1)= -2/(1-fb_opt.bat_width)+diam*verz-ballpos(1,1);
      ballpos(2,1) = -ballpos(2,1);
    end
    % Ball am rechten Rand
    if ballpos(1,1)>=1/(1-fb_opt.bat_width)-0.5*diam*verz ...
          & ballpos(2,1)>0
      ballpos(1,1) = 2/(1-fb_opt.bat_width)-diam*verz- ...
          ballpos(1,1);
      ballpos(2,1) = -ballpos(2,1);
    end
    % Ball am oberen Rand
    if ballpos(1,2)>=1-0.5*diam & ballpos(2,2)>0
      ballpos(1,2)=2-diam-ballpos(1,2);
      ballpos(2,2)= -ballpos(2,2);
    end
    
    % BAll unten am Schläger
    if ballpos(1,2)<=bahi+0.5*diam & ballpos(2,2)<0
      % 2 Fälle, Treffer oder nicht
      xpos = ballpos(1,1)+(bahi+0.5*diam-ballpos(1,2))*ballpos(2,1)/ballpos(2,2);
      if xpos>=xold-fac(2) & xpos<=xold+fac(2)
        % hit
        hit = hit+1;
        do_set(ht(1), 'string',['HIT: ', int2str(hit)]);
        do_set(xposi+20);
        if gaMo==0      
          do_set(ball,'MarkerEdgeColor',fb_opt.ball_hit_color);
          counter = round(fb_opt.interbreak_interval*fb_opt.fs/1000); 
          status = 4;
          ballpos(1,1)= xpos;ballpos(1,2) = bahi+0.5*diam;
        else
          if hit+miss>=fb_opt.matchpoints
            status = 4;
            ballpos(1,1)= xpos;ballpos(1,2) = bahi+0.5*diam;
            counter = round(fb_opt.interbreak_interval*fb_opt.fs/1000); 
          else
            ballpos(1,2)=2*bahi+diam- ballpos(1,2);
            ballpos(2,2)= -ballpos(2,2);
          end
        end
      else
        miss = miss+1;
        do_set(ball,'MarkerEdgeColor',fb_opt.ball_miss_color);
        do_set(ht(2), 'string',['MISS: ', int2str(miss)]);
        do_set(xposi+40);
        ballpos(1,1)= xpos;ballpos(1,2) = bahi+0.5*diam;
        status = 4;
        counter = round(fb_opt.interbreak_interval*fb_opt.fs/1000); 
      end
    end
    do_set(ball,'XData',ballpos(1,1),'YData',ballpos(1,2));
    if status==4
      break;
    end
    numb = numb+1;do_set('+');
  end
end

if status == 4
  % ball waits and maybe game stops
  if counter<=0
    status = 2;
    if hit+miss>=fb_opt.matchpoints
      fb_opt.status = 'stop';
      fb_opt.changed = 1;
      if fb_opt.show_bit
          a= toc;
          bit = bitrate(hit/(hit+miss), ...
			max(2,length(fb_opt.ball_positions)))/a*60*(hit+miss);
      end   
    end
  else
    counter = counter-1;
  end
end

for i = numb+1:fb_opt.repetitions
  do_set('+');
end



