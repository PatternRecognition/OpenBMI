function fb_opt = feedback_hittarget(fig, fb_opt, x, y);

persistent status counter cross cou seqpos target ht fbb target_rect hreg xpos ypos time_count free_balls stopped sequence beat played

global lost_packages

gray = [1,1,1]*0.8;
GRAY = [1,1,1]*0.8;
blue = [0 0 1];
green = [0 1 0];
red = [1 0 0];

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
                       'cross_correct_color',green,...
                       'cross_wrong_color',red,...
                       'midfield', 0.4, ...
                       'marker', 'r+', ...
                       'marker_size', [60 15], ...
                       'target_mode', 3, ...
                       'target_dist', 0.1, ...
                       'target_width', 0.05, ...
                       'balanced_sequence',1,...
                       'parPort', 1,...
                       'damping',20,...
                       'nontarget_color',gray,...
                       'target_color',blue,...
                       'show_target',[3000 5000],...
		       'relational',0,...
                       'gradient',inf,...
		       'changed',0,...
		       'log',1,...
                       'free_balls',0,...
                       'matchpoints',inf,...
                       'position', get(fig,'position'));

  if (ischar(fb_opt.sequence) & strcmp(fb_opt.sequence,'rand')) | ...
        (isnumeric(fb_opt.sequence) & fb_opt.sequence==1)
    fb_opt.sequence = 'Z';
  end

  
  active = false;
  status = 0;
  counter = ceil(fb_opt.countdown*fb_opt.fs/1000)+1;
  fb_opt.reset = 0;
  seqpos = 1;
  fbb= fb_opt;
  fbb.status = '';
  midpos = 0;

  
  free_balls = fb_opt.free_balls;
  [handle,target_rect] = feedback_hittarget_init(fig,fb_opt);
  cross = 1; cou = 2; ht = 3:4; fig = 5; gc = 6; hreg = 7:length(handle);
  if fb_opt.balanced_sequence == 1 & fb_opt.sequence=='Z' & fb_opt.matchpoints<inf

    sequence = char(48+ceil(randperm(fb_opt.matchpoints)/fb_opt.matchpoints*(length(hreg)+1)));
    sequence = [char(48+ceil(rand(1,fb_opt.free_balls)*length(hreg))),sequence];
  else
    sequence = fb_opt.sequence;
  end
  
  played = 0;
  do_set('init',handle,'hittarget',fb_opt);
  do_set(200);
  fb_opt.changed = 0;
  stopped = 0;
  if free_balls>0
    do_set(ht,'Visible','off');
  end
  beat = 0;
end




if fb_opt.changed == 1
  if fb_opt.balanced_sequence == 0
    sequence = fb_opt.balanced_sequence;
  else
    if fbb.balanced_sequence == 0 & fb_opt.sequence=='Z' & fb_opt.matchpoints<inf
      sequence = char(48+ceil(randperm(fb_opt.matchpoints-hit-miss)/(fb_opt.matchpoints-hit-miss)*(length(hreg)+1)));
    end
  end
  
  if fb_opt.score & free_balls==0
    do_set(ht(1),'Visible','on');
    do_set(ht(2),'Visible','on');
  else
    do_set(ht(1),'Visible','off');
    do_set(ht(2),'Visible','off');
  end
  
  if any(fbb.marker_size~=fb_opt.marker_size)
    do_set(cross,'MarkerSize',fb_opt.marker_size(1), ...
           'LineWidth',fb_opt.marker_size(2));
  end
  if ~strcmp(fbb.marker,fb_opt.marker)
    do_set(cross,'Marker',fb_opt.marker(2), ...
           'Color',fb_opt.marker(1));
  end
end



if fb_opt.changed | neustart   
    beat = 0;
    played = 0;
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
  do_set(cou,'String','stopped', 'Visible','on');
  do_set('+');
  return;
end



fbb= fb_opt;

  
%if status<3 & (fb_opt.relational>0 | fb_opt.middleposition>0)
%  x = 0;
%  y = 0;
if (fb_opt.relational>0)
  if fb_opt.relational==1
    x = x/fb_opt.damping+xpos;
    y = y/fb_opt.damping+ypos;
  else
    x = x*(1-fb_opt.relational)/(1-fb_opt.relational^fb_opt.damping)+xpos*fb_opt.relational;
    y = y*(1-fb_opt.relational)/(1-fb_opt.relational^fb_opt.damping)+ypos*fb_opt.relational;
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

do_set(cross, 'XData',x, 'YData',y); 



if strcmp(fb_opt.status,'pause')
  do_set(cou,'String','pause', 'Visible','on');
  do_set('+');
  return;
else

if status==0,
  do_set(cross,'Visible','off');
  do_set(ht(1),'Visible','off');
  do_set(ht(2),'Visible','off');
  counter = counter-1;
  if counter<=0
    time_count = 0;
    do_set(cou,'Visible','off');
    status = 1;
    do_set(201);
    do_set(cross,'Visible','on');
    if free_balls<=0
    do_set(ht(1),'Visible','on');
    do_set(ht(2),'Visible','on');
end
  else
    do_set(cou,'String',int2str(ceil(counter/fb_opt.fs)));
  end
end

  
if status==1
  nTargets= length(hreg);
  a = sequence(seqpos);
  seqpos = mod(seqpos,length(sequence))+1;
  if strcmp(a,'Z')
    target = ceil(rand*nTargets);
  else
    target = a-'0';
  end

  nontarget= setdiff(1:nTargets, target);
  do_set(hreg(target),'FaceColor',fb_opt.target_color);
  for i = 1:length(nontarget)
    do_set(hreg(nontarget(i)),'FaceColor',fb_opt.nontarget_color);
  end
  do_set(target);
  counter = ceil((fb_opt.show_target(1)+rand*diff(fb_opt.show_target))*fb_opt.fs/1000)+1;

  status = 2;

end


if status==2,
  touch= pointinrect([x y], target_rect);
  if ~isempty(touch) & target==touch,
    do_set(cross,'Color',fb_opt.cross_correct_color);
    if free_balls==0
      beat = beat+1;
    end
  else
    do_set(cross,'Color',fb_opt.cross_correct_color);
  end
  
  counter = counter-1;
  if counter<=0
    played = played+(free_balls==0);
    free_balls = max(0,free_balls-1);
    if free_balls<=0
      do_set(ht(2),'Visible','on');
    end
    if played>=fb_opt.matchpoints
      return;
    end
    status = 1;
  end
end
  



end

if status>0 & (free_balls==0)  
  time_count = time_count+1+lost_packages;
  do_set(ht(2),'String',sprintf('%2.1f%%',beat/time_count*100));
end




if played>=fb_opt.matchpoints
  fb_opt.status = 'stop';
  fb_opt.changed = 1;
  for i = 1:length(hreg)
    do_set(hreg(i),'FaceColor',fb_opt.nontarget_color);
  end
end


xpos = x; ypos = y;

do_set('+');