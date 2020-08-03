function fb_opt = feedback_2d_drive(fig, fb_opt, x, y);

persistent status counter active arrow pat hreg2 ...
    cou seqpos target hit miss ht fbb target_rect hreg time0 midpos ...
    do_reset fid lastbeaten xpos ypos wpos time_count timeoutcounter free_balls stopped target_next sequence pause_after_timeout pause_counter sound_files beinthemiddle beinthemiddle_count getapoint chosi success
global SOUND_DIR
global lost_packages

gray = [1,1,1]*0.8;
GRAY = [1,1,1]*0.8;
blue = [0 0 1];
green = [0 1 0];
red = [1 0 0];

if ~exist('y','var') | isempty(y)
  y = [];
end

false= 0;  %% for R<13
true= 1; %% for R<13

if isempty(fb_opt)
  fb_opt = struct('reset',1);
end

if ~isfield(fb_opt,'reset') 
  fb_opt.reset = 1;
end

neustart = 0;

if fb_opt.reset,
  neustart = 1;
  fb_opt= set_defaults(fb_opt, ...
                       'sequence', 0, ...
                       'countdown', 5000, ...
                       'fs', 25, ...
                       'status', 'pause', ...
                       'time_after_hit', 500, ...
                       'time_before_free', 500, ...
                       'free_region', 1, ...
                       'marker_active', 'r', ...
                       'marker_nonactive', 'k', ...
                       'marker_size', 0.1, ...
                       'score', 0, ...
                       'target_mode', 3, ...
                       'target_dist', 0.1, ...
                       'target_width', 0.05, ...
                       'pass_nontargets', 1, ...
                       'balanced_sequence',1,...
                       'time_constraint', 0, ...
                       'fix_speed',0,...
                       'cursor_on',1,...
                       'turn_direction',1,...
                       'reset_to_middle',1,...
                       'reset_at_edge',1,...
                       'pause_after_timeout',0,...
                       'repeat_after_timeout',0,...
                       'parPort', 1,...
                       'turn_speed',1,...
                       'acc_speed',1,...
                       'nontarget_color',gray,...
                       'show_result',1,...
                       'order_rest',[0,1],...
                       'next_target',0,...
                       'next_target_width',0,...
                       'auditory',0,...
                       'auditory_files',{'links','rechts','unten','oben'},...
                       'changed',0,...
                       'show_bit',0,...
                       'log',1,...
                       'free_balls',0,...
                       'matchpoints',inf,...
                       'position', get(fig,'position'));
  
  if length(fb_opt.order_rest)==1
    fb_opt.order_rest = [fb_opt.order_rest,1];
  end
  if (ischar(fb_opt.sequence) & strcmp(fb_opt.sequence,'rand')) | ...
        (isnumeric(fb_opt.sequence) & fb_opt.sequence==1)
    fb_opt.sequence = 'Z';
  end
  
  do_reset = 0;
  
  active = false;
  status = 0;
  counter = ceil(fb_opt.countdown*fb_opt.fs/1000)+1;
  fb_opt.reset = 0;
  seqpos = 1;
  hit = 0;
  miss = 0;
  fbb= fb_opt;
  fbb.status = '';
  midpos = 0;
  xpos = 0;
  ypos = 0;
  wpos = 0;
  
  target_next = [];
  
  free_balls = fb_opt.free_balls;
  [handle,target_rect] = feedback_2d_drive_init(fig,fb_opt);
  arrow= 1; pat = 2; cou = 3; ht = 4:5; fig = 6; gc = 7; hreg = 8:7+(length(handle)-7)/2;hreg2 = 8+(length(handle)-7)/2:length(handle);
  if fb_opt.balanced_sequence == 1 & fb_opt.sequence=='Z' & fb_opt.matchpoints<inf & fb_opt.matchpoints>0
    
    if fb_opt.order_rest(1)==0
      sequence = char(48+ceil(randperm(fb_opt.matchpoints)/fb_opt.matchpoints*length(hreg)));
      sequence = [char(48+ceil(rand(1,fb_opt.free_balls)*length(hreg))),sequence];
    else
      sequence = char(47+ceil(randperm(fb_opt.matchpoints)/fb_opt.matchpoints*(1+length(hreg))));
      sequence = [char(47+ceil(rand(1,fb_opt.free_balls)*(1+length(hreg)))),sequence];
      
    end
    
  else
    sequence = fb_opt.sequence;
  end
  
  if fb_opt.auditory
    sound_files = struct;
    for ii = 1:length(hreg);
      [sound_files(ii).sound,sound_files(ii).fs] = wavread([SOUND_DIR fb_opt.auditory_files{ii} '.wav']);
    end
  end  
  
  
  do_set('init',handle,'2d_drive',fb_opt);
  do_set(200);
  pause_after_timeout = 0;
  fb_opt.changed = 0;
  time_count = 0;
  stopped = 0;
  if free_balls>0
    do_set(ht,'Visible','off');
  end
  if ~fb_opt.cursor_on
    do_set(linie,'Visible','off');
    do_set(arrow,'Visible','off');
  end
  
  if ~fb_opt.show_result
    do_set(ht,'Visible','off');
  end
end




if fb_opt.changed == 1
% $$$   if fb_opt.balanced_sequence == 0
% $$$     sequence = fb_opt.balanced_sequence;
% $$$   else
% $$$     if fbb.balanced_sequence == 0 & fb_opt.sequence=='Z' & fb_opt.matchpoints<inf & fb_opt.matchpoints>0
% $$$       if fb_opt.order_rest(1)==0
% $$$         sequence = char(48+ceil(randperm(fb_opt.matchpoints-hit-miss)/(fb_opt.matchpoints-hit-miss)*length(hreg)));
% $$$       else
% $$$         sequence = char(47+ceil(randperm(fb_opt.matchpoints-hit-miss)/(fb_opt.matchpoints-hit-miss)*(1+length(hreg))));
% $$$       end        
% $$$     end
% $$$   end
  
  if fb_opt.score & free_balls==0 & fb_opt.show_result
    do_set(ht(1),'Visible','on');
    do_set(ht(2),'Visible','on');
  else
    do_set(ht(1),'Visible','off');
    do_set(ht(2),'Visible','off');
  end
  
  if fbb.free_region~=fb_opt.free_region,
    if fb_opt.free_region>=1,
      do_set(pat,'Visible','off');
    else
      do_set(pat, 'XData',[-1 1 1 -1]*fb_opt.free_region, ...
             'YData',[-1 -1 1 1]*fb_opt.free_region, ...
             'Visible','on');
    end
  end
  
  if fbb.cursor_on~=fb_opt.cursor_on
    if fb_opt.cursor_on
      do_set(linie,'Visible','on');
      do_set(arrow,'Visible','on');
    else  
      do_set(linie,'Visible','off');
      do_set(arrow,'Visible','off');
    end
  end
  
  if fbb.show_result~=fb_opt.show_result
    if fb_opt.show_result & fb_opt.score
      do_set(ht,'Visible','on');
    else
      do_set(ht,'Visible','off');
    end
  end
  
  if fbb.next_target~=fb_opt.next_target
    if fb_opt.next_target==1
      do_set(hreg2,'Visible','on');
    else
      do_set(hreg2,'Visible','off');
    end
  end
end



if fb_opt.changed | neustart   
  if ~strcmp(fbb.status,fb_opt.status)
    switch fb_opt.status
     case 'play'
      do_set(210);
     case 'pause'
      if pause_after_timeout==0
        do_set(211);
      end
     case 'stop'
      do_set(212);
      stopped = 1;
    end
    switch fb_opt.status
     case 'play'
      if stopped
        stopped = 0;
        time_count = 0;
        hit = 0; miss = 0;
        do_set(ht(2), 'string','MISS: 0');
        do_set(ht(1), 'string','HIT: 0');
        status = 0;
        counter = ceil(fb_opt.countdown*fb_opt.fs/1000)+1;
        seqpos = 1;
      end
      if status>0
        do_set(cou,'Visible','off');
      end 
     case  'stop'
      fbb.status = 'stop';
    end
  end
end


fb_opt.changed = 0;




if strcmp(fb_opt.status, 'stop')
  if fb_opt.show_bit==1
    if (hit+miss)>0
      bi = bitrate(hit/(hit+miss),size(target_rect,1)+(fb_opt.order_rest(1)>0));
    else
      bi = 0;
    end
    if time_count~=0
      tim = time_count/fb_opt.fs;
      bi = bi*(hit+miss)/tim*60;
      minu = floor(tim/60);
      sec = round(tim-minu*60);
      do_set(cou,'String',sprintf('%2.1f bit/min (%d''%02d)',bi,minu,sec),'Visible','on');
      time_count = 0;
      
    end
    if fb_opt.score
      do_set(ht,'Visible','on');
    end
  else
    do_set(cou,'String','stopped', 'Visible','on');
  end
  do_set('+');
  return;
end



fbb= fb_opt;


x_arrow = [-1 0.5 0.5 1 0.5 0.5 -1 -1]*fb_opt.marker_size;
y_arrow = [-0.1,-0.1,-0.5,0,0.5,0.1,0.1,-0.1]*fb_opt.marker_size;

turn = 0;
acc = 0;

if do_reset & fb_opt.reset_at_edge
    xpos = 0;
    ypos = 0;
end

do_reset = 0;

if isempty(y)
  % 2 Klassen Steuerung
  if x<0 % turn around
    turn = -x*fb_opt.turn_direction;
  else % accelerate
    acc = x;
  end
else
  if x<0 & y<0
    acc = -max(x,y);
  elseif x>y
    turn = (x-max(0,y))*fb_opt.turn_direction;
  else
    turn = -(y-max(0,x))*fb_opt.turn_direction;
  end
end
if status<3 & fb_opt.reset_to_middle
  acc = 0;
  x = 0;
  xpos = 0;
  ypos = 0;
  y = 0;
end

if fb_opt.fix_speed
  turn = sign(turn);
  acc = sign(acc);
end

      
turn = turn*fb_opt.turn_speed;
acc = acc*fb_opt.acc_speed;

w = mod(wpos+turn,2*pi);

x = xpos+acc*cos(w);
y = ypos+acc*sin(w);

z = exp(complex(0,1)*w)*complex(x_arrow,y_arrow);
x_arrow = real(z);
y_arrow = imag(z);


if any(abs(x+x_arrow)>1);
  acce = sign(x)*(max(abs(x+x_arrow))-1);
  x = x-acce;
  y = y-acce*tan(w);
  do_reset = 1;
end


if any(abs(y+y_arrow)>1);
    acce = sign(y)*(max(abs(y+y_arrow))-1);
    y = y-acce;
    x = x-acce*cot(w);
    do_reset = 1;
end

do_set(arrow, 'XData',x+x_arrow, 'YData',y+y_arrow); 
x_spitze = x+x_arrow(4);
y_spitze = y+y_arrow(4);


if strcmp(fb_opt.status,'pause')
  if pause_after_timeout==0
    do_set(cou,'String','pause', 'Visible','on');
  else
    pause_counter = pause_counter-1;
    do_set(cou,'String',int2str(ceil(pause_counter/fb_opt.fs)), 'Visible','on');
    if pause_counter<=0
      fb_opt.status = 'play';
      fb_opt.changed = 1;
      do_set(250);
      if fb_opt.auditory
        wavplay(sound_files(target).sound,sound_files(target).fs,'async');
      else  
        do_set(hreg(target),'FaceColor',blue);
      end
      pause_after_timeout = 0;
      timeoutcounter = ceil(fb_opt.time_constraint*fb_opt.fs/1000);
      status = 3;
      do_set(cou,'Visible','off');
    end
  end
  do_set('+');
else
  
  if status==0,
    counter = counter-1;
    if counter<=0
      time_count = 0;
      do_set(cou,'Visible','off');
      status = 1;
      counter = ceil(fb_opt.time_after_hit*fb_opt.fs/1000)+1;
      do_set(201);
    else
      do_set(cou,'String',int2str(ceil(counter/fb_opt.fs)));
    end
  end
  
  if status==1,
    lastbeaten = 0;
    counter = counter-1;
    if counter<=0
      if isnumeric(sequence)
        for i = 1:length(hreg)
          do_set(hreg(i),'FaceColor',fb_opt.nontarget_color);
          do_set(hreg2(i),'FaceColor',fb_opt.nontarget_color);
        end
        target = 0;
      else
        nTargets= length(hreg)+(fb_opt.order_rest(1)>0);
        a = sequence(seqpos);
        seqpos = mod(seqpos,length(sequence))+1;
        if strcmp(a,'Z')
          target = ceil(rand*nTargets)-(fb_opt.order_rest(1)~=0);
        else
          target = a-'0';
        end
        
        if fb_opt.next_target 
          if isempty(target_next)
            a = sequence(seqpos);
            seqpos = mod(seqpos,length(sequence))+1;
            if strcmp(a,'Z')
              target_next = ceil(rand*nTargets)-(fb_opt.order_rest(1)==0);
            else
              target_next = a-'0';
            end
          else
            hhhh = target;
            target= target_next;
            target_next = hhhh;
          end
          
          
          if fb_opt.order_rest(1)>0 
            nontarget_next= setdiff(1:nTargets-1, target_next);
            nontarget= setdiff(1:nTargets-1, target);
          else    
            nontarget_next= setdiff(1:nTargets, target_next);
            nontarget= setdiff(1:nTargets, target);
          end    
          if fb_opt.matchpoints>0 & hit+miss+1>=fb_opt.matchpoints
            nontarget_next= 1:nTargets;
          else
            if target_next>0
              do_set(hreg2(target_next),'FaceColor',blue);
            end
          end
          
          if fb_opt.auditory
            if target>0
              wavplay(sound_files(target).sound,sound_files(target).fs,'async');
            end
          else 
            if target>0
              do_set(hreg(target),'FaceColor',blue);
            else
              do_set(pat,'FaceColor',blue);
            end
          end
          for i = 1:length(nontarget)
            do_set(hreg(nontarget(i)),'FaceColor',fb_opt.nontarget_color);
            if fb_opt.order_rest>0 & target>0
              do_set(pat,'FaceColor',fb_opt.nontarget_color);
            end
          end
          for i = 1:length(nontarget_next)
            do_set(hreg2(nontarget_next(i)),'FaceColor',fb_opt.nontarget_color);
          end
          do_set(target+40*(target==0));
        else
          if fb_opt.order_rest(1)>0
            nontarget = setdiff(1:nTargets-1,target);
          else
            nontarget= setdiff(1:nTargets, target);
          end
          if fb_opt.auditory
            if target>0
              wavplay(sound_files(target).sound,sound_files(target).fs,'async');
            end
          else 
            if target>0
              do_set(hreg(target),'FaceColor',blue);
            else
              do_set(pat,'FaceColor',blue);
            end          
          end
          for i = 1:length(nontarget)
            do_set(hreg(nontarget(i)),'FaceColor',fb_opt.nontarget_color);
            if fb_opt.order_rest>0 & target>0
              do_set(pat,'FaceColor',fb_opt.nontarget_color);
            end
          end
          do_set(target+40*(target==0));
        end
      end
      
      status = 2;
      counter = ceil(fb_opt.time_before_free*fb_opt.fs/1000);
      if fb_opt.time_constraint>0
        timeoutcounter = ceil(fb_opt.time_constraint*fb_opt.fs/1000);
      end
      
    end
  end
  
  if status==2,
    counter = counter-1;
    if counter<=0
      status = 3;
    else
      if fb_opt.time_constraint>0, timeoutcounter = timeoutcounter-1;end
    end
  end
  
  
  
  if status==3,
    
    if (fb_opt.free_region>=1) | (max(abs(x_spitze),abs(y_spitze))<fb_opt.free_region)  | ((abs(x_spitze)<fb_opt.free_region) & ((y_spitze*ypos)<=0)) ...
          | ((abs(y_spitze)<fb_opt.free_region) & ((x_spitze*xpos)<=0)) | (((x_spitze*xpos)<=0) & ((y_spitze*ypos)<=0))
      status = 4;
      active = true;
      do_set(60);
      do_set(arrow, 'FaceColor',fb_opt.marker_active);
      if target==0 & fb_opt.order_rest(1)>0
        beinthemiddle = fb_opt.order_rest(1)/fb_opt.fs;
        beinthemiddle_count = [0,0];
      end
    else
      if fb_opt.time_constraint>0, timeoutcounter = timeoutcounter-1;if timeoutcounter<0; status = 4; end; end
    end
  end
  
  if status==4,
    leaveit = 0;
    if fb_opt.time_constraint>0
      timeoutcounter = timeoutcounter-1;
      if timeoutcounter<0
        if ~fb_opt.repeat_after_timeout
          do_set(hreg(target), 'FaceColor',red);
          status= 1;
          do_set(arrow,'FaceColor',fb_opt.marker_nonactive);
          
          counter = ceil(fb_opt.time_after_hit*fb_opt.fs/1000);
          active= false;
          miss = miss+1;
          
          do_set(ht(2), 'string',['MISS: ' int2str(miss)]);
          do_set(40+target);
          leaveit = 1;
        else
          do_set(hreg,'FaceColor',fb_opt.nontarget_color);
          do_set(arrow,'FaceColor',fb_opt.marker_nonactive);
          active= false;
          if fb_opt.pause_after_timeout>0
            pause_counter = ceil(fb_opt.pause_after_timeout*fb_opt.fs/1000);
            fb_opt.status = 'pause';
            pause_after_timeout = 1;
            do_set(249);
            fb_opt.changed = 1;
          else
            status = 1;
            counter = ceil(fb_opt.time_after_hit*fb_opt.fs/1000);                  
            do_set(50+target);
          end   
          leaveit = 1;
        end
      end
    end
    
    if leaveit == 0
      if fb_opt.order_rest(1)>0 & beinthemiddle>0 & target==0
        touch = pointinrect([x_spitze,y_spitze],[-1 -1 1 1]*fb_opt.free_region);
        if isempty(touch)
          do_set(arrow,'FaceColor',fb_opt.marker_nonactive);
        else
          do_set(arrow,'FaceColor',fb_opt.marker_active);
        end
        
        beinthemiddle_count= beinthemiddle_count+[~isempty(touch),1];
        beinthemiddle = beinthemiddle-1;
        if beinthemiddle<=0
          beinthemiddle_count = beinthemiddle_count(1)/beinthemiddle_count(2)>=fb_opt.order_rest(2);
          if beinthemiddle_count
            if fb_opt.show_result  
              do_set(pat, 'FaceColor',chosen);
              chosi = 0;
            else
              do_set(pat, 'FaceColor',fb_opt.nontarget_color);
            end
            if free_balls>0
              free_balls=free_balls-1;
              if free_balls==0 & fb_opt.score
                do_set(ht,'Visible','on');
              end
              getapoint = 0;
            else
              getapoint = 1;
            end
            do_set(50);
          else
            if fb_opt.show_result
              %if fb_opt.reject_mode==0
              do_set(pat, 'FaceColor',red);
              %else
              % do_set(hreg,'FaceColor',chosen);
              %chosi = -1;
              %end
            else  
              do_set(pat, 'FaceColor',fb_opt.nontarget_color);
            end
            if free_balls>0
              free_balls=free_balls-1;
              if free_balls==0 & fb_opt.score
                do_set(ht,'Visible','on');
              end
              getapoint = 0;
            else
              getapoint = -1;
            end
            do_set(51);
          end
          status = 1;
          do_set(arrow,'FaceColor',fb_opt.marker_nonactive);
          
          counter = ceil(fb_opt.time_after_hit*fb_opt.fs/1000);
          active = false;
          
        end
      else
        touch= pointinrect([x_spitze y_spitze], target_rect);
        if ~isempty(touch),
          if target==0,
            if fb_opt.show_result  
              do_set(hreg(touch), 'FaceColor',red);          
            else
              do_set(hreg(touch), 'FaceColor',fb_opt.nontarget_color);
            end
            chosi = touch;
            do_set(30+touch);
          else      
            if target == touch,
              if fb_opt.show_result
                do_set(hreg(touch), 'FaceColor',green); % war chosen
              else
                do_set(hreg(touch), 'FaceColor',fb_opt.nontarget_color);
              end 
              if free_balls>0
                free_balls=free_balls-1;
                if free_balls==0 & fb_opt.score & fb_opt.show_result
                  do_set(ht,'Visible','on');
                end
              else
                hit = hit+1;
              end
              do_set(10+touch);
            else
              if ~fb_opt.pass_nontargets,
                do_set(hreg, 'FaceColor',fb_opt.nontarget_color);
                if fb_opt.show_result
                  do_set(hreg(touch), 'FaceColor',red); % war chosen
                end   
                if free_balls>0
                  free_balls=free_balls-1;
                  if free_balls==0 & fb_opt.score
                    do_set(ht,'Visible','on');
                  end
                else
                  miss = miss+1;
                end
                
              end
              if lastbeaten ~= touch
                do_set(20+touch);
                lastbeaten = touch;
              end
            end
            do_set(ht(2), 'string',['MISS: ' int2str(miss)]);
            do_set(ht(1), 'string',['HIT: ', int2str(hit)]);

          end
          
          if (target==touch) | ~fb_opt.pass_nontargets | target==0
            status = 1;
            do_set(arrow,'FaceColor',fb_opt.marker_nonactive);
            
            counter = ceil(fb_opt.time_after_hit*fb_opt.fs/1000);
            active = false;
            do_reset = 0;
          end
        else
          lastbeaten = 0;
        end
      end
    end
  end
  
  
  if status>0 & (free_balls==0)  
    time_count = time_count+1+lost_packages;
  end
  
  
  
end

if ((fb_opt.matchpoints>0 & hit+miss>=fb_opt.matchpoints) | (fb_opt.matchpoints<0 & time_count/fb_opt.fs>abs(fb_opt.matchpoints)))
  fb_opt.status = 'stop';
  fb_opt.changed = 1;
  if fb_opt.score
    do_set(ht,'Visible','on');
  end
end


xpos = x; ypos = y; wpos = w;

do_set('+');