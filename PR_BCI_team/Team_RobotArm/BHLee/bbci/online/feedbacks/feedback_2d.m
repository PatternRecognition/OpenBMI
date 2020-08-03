function fb_opt = feedback_2d(fig, fb_opt, x, y,reject);

persistent status counter active cross pat hreg2 ...
    cou seqpos target hit miss ht fbb target_rect hreg time0 midpos ...
    fid lastbeaten xpos ypos time_count timeoutcounter free_balls stopped target_next sequence pause_after_timeout pause_counter sound_files beinthemiddle beinthemiddle_count getapoint chosi success
global SOUND_DIR
global lost_packages

gray = [1,1,1]*0.8;
GRAY = [1,1,1]*0.8;
blue = [0 0 1];
green = [0 1 0];
red = [1 0 0];
chosen = [1 0.7 0];
notchosen = [0.5 0.5 0.5];


if ~exist('y','var') | isempty(y)
  y = 0;
end

if ~exist('reject','var') | isempty(reject)
  reject = -1; % accept everything
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
                       'marker_active', 'r+', ...
                       'marker_nonactive', 'k.', ...
                       'marker_active_size', [60 15], ...
                       'marker_nonactive_size', [50 5], ...
                       'score', 0, ...
                       'target_mode', 3, ...
                       'target_dist', 0.1, ...
                       'target_width', 0.05, ...
                       'pass_nontargets', 1, ...
                       'balanced_sequence',1,...
                       'time_constraint', 0, ...
                       'cursor_on',1,...
                       'star',0,...
                       'pause_after_timeout',0,...
                       'repeat_after_timeout',0,...
                       'parPort', 1,...
                       'damping',20,...
                       'nontarget_color',gray,...
                       'relational',0,...
                       'show_result',1,...
                       'gradient',inf,...
                       'order_rest',[0,1],...
                       'next_target',0,...
                       'fix_lines',0,...
                       'next_target_width',0,...
                       'auditory',0,...
                       'auditory_files',{'links','rechts','unten','oben'},...
                       'middleposition',0,...
                       'changed',0,...
                       'show_bit',0,...
                       'log',1,...
                       'reject_mode',0,...  % 0: accept everything, 1: reject and repeat, 2: choose other if it makes sense
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
  
  target_next = [];
  
  free_balls = fb_opt.free_balls;
  [handle,target_rect] = feedback_2d_init(fig,fb_opt);
  cross = 1; pat = 2; cou = 3; ht = 4:5; fig = 6; gc = 7; hreg = 8:7+(length(handle)-7)/2;hreg2 = 8+(length(handle)-7)/2:length(handle);
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
  
  
  do_set('init',handle,'2d',fb_opt);
  do_set(200);
  pause_after_timeout = 0;
  fb_opt.changed = 0;
  time_count = 0;
  stopped = 0;
  if free_balls>0
    do_set(ht,'Visible','off');
  end
  if ~fb_opt.cursor_on
    do_set(cross,'Visible','off');
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
      do_set(cross,'Visible','on');
    else  
      do_set(cross,'Visible','off');
    end
  end
  
  if fbb.show_result~=fb_opt.show_result
    if fb_opt.show_result & fb_opt.score
      do_set(ht,'Visible','on');
    else
      do_set(ht,'Visible','off');
    end
  end
  
  if active,
    if any(fbb.marker_active_size~=fb_opt.marker_active_size)
      do_set(cross,'MarkerSize',fb_opt.marker_active_size(1), ...
             'LineWidth',fb_opt.marker_active_size(2));
    end
    if ~strcmp(fbb.marker_active,fb_opt.marker_active)
      do_set(cross,'Marker',fb_opt.marker_active(2), ...
             'Color',fb_opt.marker_active(1));
    end
  else
    if any(fbb.marker_nonactive_size~=fb_opt.marker_nonactive_size)
      do_set(cross,'MarkerSize',fb_opt.marker_nonactive_size(1), ...
             'LineWidth',fb_opt.marker_nonactive_size(2));
    end
    if ~strcmp(fbb.marker_nonactive,fb_opt.marker_nonactive)
      do_set(cross,'Marker',fb_opt.marker_nonactive(2), ...
             'Color',fb_opt.marker_nonactive(1));
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

if (status<3 | status==5) & (fb_opt.relational>0 | fb_opt.middleposition>0)
  x = 0;
  y = 0;
else
  if fb_opt.middleposition &midpos>0
    x = (fb_opt.middleposition-midpos)/fb_opt.middleposition*x;
    y=  (fb_opt.middleposition-midpos)/fb_opt.middleposition*y;
    midpos = midpos-1;
  end
  
  if fb_opt.fix_lines
    switch fb_opt.target_mode
     case 1
      y = 0;
     case 2
      x = 0;
     case 3
      if abs(x)>abs(y)
        x = sign(x)*(abs(x)-abs(y));
        y = 0;
      else
        y = sign(y)*(abs(y)-abs(x));
        x = 0;
      end
      if fb_opt.relational>0
        if xpos*ypos~=0,
          if abs(xpos)>abs(ypos)
            xpos = sign(xpos)*(abs(xpos)-abs(ypos));
            ypos = 0;
          else
            ypos = sign(ypos)*(abs(ypos)-abs(xpos));
            xpos = 0;
          end
        end
        if xpos==0
          if abs(x)>0
            if fb_opt.relational==1
              rel = x/fb_opt.damping;
            else    
              rel = x*(1-fb_opt.relational)/(1-fb_opt.relational^fb_opt.damping);
            end
            if rel>abs(x)*fb_opt.relational
              y = 0;
              x = sign(rel)*(rel-abs(xpos)*fb_opt.relational);
            else
              y = sign(ypos)*(abs(ypos)*fb_opt.relational-rel);
            end
          else
            if fb_opt.relational==1
              rel = y/fb_opt.damping;
            else    
              rel = y*(1-fb_opt.relational)/(1-fb_opt.relational^fb_opt.damping);
            end
            y = ypos*fb_opt.relational + rel;
          end
        else
          if abs(x)>0
            if fb_opt.relational==1
              rel = x/fb_opt.damping;
            else    
              rel = x*(1-fb_opt.relational)/(1-fb_opt.relational^fb_opt.damping);
            end
            x = xpos*fb_opt.relational + rel;
          else
            if fb_opt.relational==1
              rel = y/fb_opt.damping;
            else    
              rel = y*(1-fb_opt.relational)/(1-fb_opt.relational^fb_opt.damping);
            end
            if rel>abs(y)*fb_opt.relational
              x = 0;
              y = sign(rel)*(rel-abs(xpos)*fb_opt.relational);
            else
              x = sign(ypos)*(abs(ypos)*fb_opt.relational-rel);
            end
          end                           
        end
      end        
      
      
      
     case 8
      if fb_opt.star
        [dum,ric] = sort([x,y,0]);
        ri = dum(3)-dum(2);
        ric = ric(3);
      else
        if x==0
          if y>0
            wi = pi/2;
          else
            wi = 3*pi/2;
          end
        else
          wi = atan(abs(y/x));
          if y>0 & x<0
            wi = pi-wi;
          elseif y<0 & x<0
            wi = pi+wi;
          elseif y<0 & x>0
            wi = 2*pi-wi;
          end
        end
        
        if  wi>=0 & wi<2/3*pi
          ric = 2;
          ri = sqrt(x.^2+y.^2)*abs(wi-pi/3);
        elseif 2/3*pi<=wi & wi<4/3*pi
          ric = 3;
          ri = sqrt(x.^2+y.^2)*abs(wi-pi);
        else 
          ric = 1;
          ri = sqrt(x.^2+y.^2)*abs(wi-5/6*pi);
        end
      end
      
      if fb_opt.relational>0
        ri_old = sqrt(xpos^2+ypos^2);
        if ypos==0
          ric_old = 3;
        elseif ypos>0
          ric_old = 2;
        else
          ric_old = 1;
        end
        if fb_opt.relational<1
          ri = ri*(1-fb_opt.relational)/(1-fb_opt.relational^fb_opt.damping);
        else
          ri = ri/fb_opt.damping;
        end
        if ric==ric_old
          ri = ri+ri_old*fb_opt.relational;
        else
          ri = ri_old*fb_opt.relational-ri;
          if ri<0 
            ri = abs(ri);
          else
            ric = ric_old;
          end
        end
      end
      
      switch ric
       case 1
        x = ri*cos(pi/3);
        y = -ri*sin(pi/3);
       case 2
        x = ri*cos(pi/3);
        y = ri*sin(pi/3);
       case 3
        x = -ri;
        y = 0;
      end
      
      
      
      
     case 9
      if fb_opt.star
        [dum,ric] = sort([x,y,0]);
        ri = dum(3)-dum(2);
        ric = ric(3);
      else
        if x==0
          if y>0
            wi = pi/2;
          else
            wi = 3*pi/2;
          end
        else
          wi = atan(abs(y/x));
          if y>0 & x<0
            wi = pi-wi;
          elseif y<0 & x<0
            wi = pi+wi;
          elseif y<0 & x>0
            wi = 2*pi-wi;
          end
        end
        if pi/3<=wi & wi<pi
          ric = 2;
          ri = sqrt(x.^2+y.^2)*abs(wi-2*pi/3);
        elseif pi<=wi & wi<5/6*pi
          ric = 1;
          ri = sqrt(x.^2+y.^2)*abs(wi-4/3*pi);
        else 
          ric = 3;
          if wi>0
            ri = sqrt(x.^2+y.^2)*wi;
          else
            ri = sqrt(x^2+y^2)*(2*pi-wi);
          end
        end
      end
      
      if fb_opt.relational>0
        ri_old = sqrt(xpos^2+ypos^2);
        if ypos==0
          ric_old = 3;
        elseif ypos>0
          ric_old = 2;
        else
          ric_old = 1;
        end
        if fb_opt.relational<1
          ri = ri*(1-fb_opt.relational)/(1-fb_opt.relational^fb_opt.damping);
        else
          ri = ri/fb_opt.damping;
        end
        if ric==ric_old
          ri = ri+ri_old*fb_opt.relational;
        else
          ri = ri_old*fb_opt.relational-ri;
          if ri<0 
            ri = abs(ri);
          else
            ric = ric_old;
          end
        end
      end
      
      switch ric
       case 1
        x = -ri*cos(pi/3);
        y = -ri*sin(pi/3);
       case 2
        x = -ri*cos(pi/3);
        y = ri*sin(pi/3);
       case 3
        x = ri;
        y = 0;
      end
      
     otherwise
      error('error');
      
    end
    
  else
    if fb_opt.star & fb_opt.target_mode ==8
      z = [x;y;0]; z = z-min(z);
      z = [cot(pi/3),-1; cot(pi/3),1; -1 0]'*z;
      x = z(1);
      y = z(2);
    end
    if fb_opt.star & fb_opt.target_mode ==9
      z = [x;y;0]; z = z-min(z);
      z = [-cot(pi/3),-1; -cot(pi/3),1; 1 0]'*z;
      x = z(1);
      y = z(2);
    end
    if (fb_opt.relational>0 | fb_opt.middleposition>0)
      
      if fb_opt.relational==1
        x = x/fb_opt.damping+xpos;
        y = y/fb_opt.damping+ypos;
      else
        x = x*(1-fb_opt.relational)/(1-fb_opt.relational^fb_opt.damping)+xpos*fb_opt.relational;
        y = y*(1-fb_opt.relational)/(1-fb_opt.relational^fb_opt.damping)+ypos*fb_opt.relational;
      end
    end
    
  end

  if abs(x)>1
    y = y/abs(x);
    x = x/abs(x);
  end
  if abs(y)>1
    x = x/abs(y);
    y = y/abs(y);
  end
  
  if x-xpos>fb_opt.gradient
    x = xpos+fb_opt.gradient;
  elseif xpos-x>fb_opt.gradient
    x = xpos-fb_opt.gradient;
  end
  
  if y-ypos>fb_opt.gradient
    y = ypos+fb_opt.gradient;
  elseif ypos-y>fb_opt.gradient
    y = ypos-fb_opt.gradient;
  end
end

do_set(cross, 'XData',x, 'YData',y); 



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
  return;
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
    
    if (fb_opt.free_region>=1) | (max(abs(x),abs(y))<fb_opt.free_region)  | ((abs(x)<fb_opt.free_region) & ((y*ypos)<=0)) ...
          | ((abs(y)<fb_opt.free_region) & ((x*xpos)<=0)) | (((x*xpos)<=0) & ((y*ypos)<=0))
      status = 4;
      active = true;
      do_set(60);
      do_set(cross,'MarkerSize',fb_opt.marker_active_size(1), ...
             'LineWidth',fb_opt.marker_active_size(2), ...
             'Marker',fb_opt.marker_active(2), ...
             'Color',fb_opt.marker_active(1));
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
          do_set(cross,'Marker',fb_opt.marker_nonactive(2), ...
                 'Color',fb_opt.marker_nonactive(1), ...
                 'MarkerSize',fb_opt.marker_nonactive_size(1), ...
                 'LineWidth',fb_opt.marker_nonactive_size(2));
          
          counter = ceil(fb_opt.time_after_hit*fb_opt.fs/1000);
          active= false;
          miss = miss+1;
          
          do_set(ht(2), 'string',['MISS: ' int2str(miss)]);
          do_set(40+target);
          midpos = fb_opt.middleposition;
          leaveit = 1;
        else
          do_set(hreg,'FaceColor',fb_opt.nontarget_color);
          do_set(cross,'Marker',fb_opt.marker_nonactive(2), ...
                 'Color',fb_opt.marker_nonactive(1), ...
                 'MarkerSize',fb_opt.marker_nonactive_size(1), ...
                 'LineWidth',fb_opt.marker_nonactive_size(2));
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
            midpos = fb_opt.middleposition;
          end   
          leaveit = 1;
        end
      end
    end
    
    if leaveit == 0
      if fb_opt.order_rest(1)>0 & beinthemiddle>0 & target==0
        touch = pointinrect([x,y],[-1 -1 1 1]*fb_opt.free_region);
        if isempty(touch)
          do_set(cross,'Marker',fb_opt.marker_nonactive(2), ...
                 'Color',fb_opt.marker_nonactive(1), ...
                 'MarkerSize',fb_opt.marker_nonactive_size(1), ...
                 'LineWidth',fb_opt.marker_nonactive_size(2));
        else
          do_set(cross,'Marker',fb_opt.marker_active(2), ...
                 'Color',fb_opt.marker_active(1), ...
                 'MarkerSize',fb_opt.marker_active_size(1), ...
                 'LineWidth',fb_opt.marker_active_size(2));
          
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
            midpos = fb_opt.middleposition;
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
          status = 5;
          do_set(cross,'Marker',fb_opt.marker_nonactive(2), ...
                 'Color',fb_opt.marker_nonactive(1), ...
                 'MarkerSize',fb_opt.marker_nonactive_size(1), ...
                 'LineWidth',fb_opt.marker_nonactive_size(2));
          
          counter = ceil(fb_opt.time_after_hit*fb_opt.fs/1000);
          active = false;
          
        end
      else
        touch= pointinrect([x y], target_rect);
        if ~isempty(touch),
          if target==0,
            if fb_opt.show_result  
              do_set(hreg(touch), 'FaceColor',chosen);          
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
              chosi = touch;
              success = green;
              if free_balls>0
                free_balls=free_balls-1;
                if free_balls==0 & fb_opt.score & fb_opt.show_result
                  do_set(ht,'Visible','on');
                end
                getapoint = 0;
              else
                getapoint = 1;
              end
              do_set(10+touch);
              midpos = fb_opt.middleposition;
            else
              if ~fb_opt.pass_nontargets,
                if fb_opt.show_result
                  do_set(hreg(touch), 'FaceColor',red); % war chosen
                else
                  do_set(hreg(touch), 'FaceColor',fb_opt.nontarget_color);
                end   
                chosi = touch;
                success = red;
                if free_balls>0
                  free_balls=free_balls-1;
                  if free_balls==0 & fb_opt.score
                    do_set(ht,'Visible','on');
                  end
                  getapoint = 0;
                else
                  getapoint = -1;
                end
                
              end
              if lastbeaten ~= touch
                do_set(20+touch);
                lastbeaten = touch;
              end
            end
          end
          
          if (target==touch) | ~fb_opt.pass_nontargets | target==0
            status = 5;
            do_set(cross,'Marker',fb_opt.marker_nonactive(2), ...
                   'Color',fb_opt.marker_nonactive(1), ...
                   'MarkerSize',fb_opt.marker_nonactive_size(1), ...
                   'LineWidth',fb_opt.marker_nonactive_size(2));
            
            counter = ceil(fb_opt.time_after_hit*fb_opt.fs/1000);
            active = false;
          end
        else
          lastbeaten = 0;
        end
      end
    end
  end
  
  if status == 5
    switch fb_opt.reject_mode
     case 0
      let_it = 1;
     case 1
      if reject<0
        let_it = 1;
        do_set(71);
      elseif reject>0
        let_it = 3;
        do_set(73);
      else
        let_it = 0;
      end
     case 2
      if reject<0
        let_it = 1;
        do_set(71);
      elseif reject>0
        if chosi == -1
          let_it = 2;
          do_set(72);
          take = 0;
        else
          if size(target_rect,1)>2
            let_it = 3;
            do_set(73);
          else
            if chosi == 0
              let_it = 3;
              do_set(73);
            else
              let_it = 2;
              do_set(72);
              take = 3-chosi;
            end
          end
        end
      else
        let_it = 0;
      end
    end
    switch let_it
     case 0
      % nothing
     case 1
      if target==0
        do_set(pat,'FaceColor',fb_opt.nontarget_color);
      else
        do_set(hreg(target), 'FaceColor',fb_opt.nontarget_color);
      end
      switch getapoint
       case +1
        hit = hit+1;
       case -1
        miss = miss+1;
       case 0
        % nothing;
      end
      
      if fb_opt.show_result
        if chosi==-1
          do_set(hreg,'FaceColor',fb_opt.nontarget_color);
          do_set(pat,'FaceColor',red);
        elseif chosi==0
          do_set(pat,'FaceColor',green);
        else
          do_set(hreg(chosi),'FaceColor',success);
        end
      end
      status=1;
     case 2
      % change it
      if target==0
        do_set(pat,'FaceColor',fb_opt.nontarget_color);
      else
        do_set(hreg(target), 'FaceColor',fb_opt.nontarget_color);
      end
      switch getapoint
       case -1
        hit = hit+1;
       case +1
        miss = miss+1;
       case 0
        % nothing;
      end
      do_set(hreg,'FaceColor',fb_opt.nontarget_color);
      do_set(pat,'FaceColor',fb_opt.nontarget_color);
      if take==0
        do_set(pat,'FaceColor',red+green-success);
      else
        do_set(hreg(take),'FaceColor',red+green-success);
      end
      status = 1;
      
     case 3
      % repeat it
      getapoint = 0;
      if fb_opt.show_result
        if chosi==-1
          do_set(hreg,'FaceColor',fb_opt.nontarget_color);
          do_set(pat,'FaceColor',notchosen);
        elseif chosi==0
          do_set(pat,'FaceColor',notchosen);
        else
          do_set(hreg(chosi),'FaceColor',notchosen);
        end
      end
      status=1;
      
      
    end
    
    
    do_set(ht(2), 'string',['MISS: ' int2str(miss)]);
    do_set(ht(1), 'string',['HIT: ', int2str(hit)]);
    
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


xpos = x; ypos = y;

do_set('+');