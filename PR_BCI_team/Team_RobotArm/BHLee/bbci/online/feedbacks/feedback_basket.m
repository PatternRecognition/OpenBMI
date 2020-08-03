function opt = feedback_basket(fig, opt, control,dum);

persistent state cursor target paused sequence h_rect2
persistent basket counter running_time waiting moveIdx moveStep
persistent h_cursor h_cross h_counter h_text h_rect free_balls next_target

global lost_packages

if ~isfield(opt,'reset') 
  opt.reset = 1;
end

if opt.reset,
  opt= set_defaults(opt, ...
                    'trial_duration', 2000, ...
                    'targets', 4, ...
                    'target_height', 0.1, ...
                    'outerTarget_size', 0.5, ...
                    'orientation', 'landscape', ...
                    'direction', 'downward', ...
                    'relative', 0, ...
                    'cone', 0, ...
                    'cursor_size', 30, ...
                    'cursor_type', 'o', ...
                    'cursor_color', [0.8 0 0.8], ...
                    'counter_pos', [-0.86 -0.9], ...
                    'counter_fontsize', 0.06, ...
                    'msg_fontsize', 0.15, ...
                    'show_counter', 'on', ...
                    'show_fixation', 'off', ...
                    'background_color', 0.9*[1 1 1], ...
                    'target_color', [0 0.7 0], ...
                    'nontarget_color', 0.4*[1 1 1], ...
                    'hit_color', [0 0.9 0], ...
                    'miss_color', [1 0 0], ...
                    'fixation_color', 0.3*[1 1 1], ...
                    'fixation_size', 0.075, ...
                    'fixation_linewidth', 3, ...
                    'fixation_position', [0 0], ...
                    'free_balls',0,...
                    'time_after_hit', 250, ...
                    'time_before_next', 0, ...
                    'time_before_free', 750, ...
                    'next_target',0,...
                    'next_target_width',0.2,...
                    'countdown', 5000, ...
                    'matchpoints', 100, ...
                    'show_points', 'on', ...
                    'ms', 40, ...
                    'balanced_sequence',0,...
                    'damping', 10, ...
                    'status', 'pause', ...
                    'log', 0, ...
                    'changed', 0, ...
                    'position', get(fig,'position'),...
                    'parPort',1);

  [handle,other_args]=feedback_basket_init(fig,opt);
  do_set('init',handle,'basket',opt);

  
  % These are the handle indices according to feedback_basket_init:
  h_cursor = 3;
  h_cross = 4:5;
  h_counter = 6;
  h_text = 7;
  h_rect = 8:7+0.5*(length(handle)-7);
  h_rect2 = 8+0.5*(length(handle)-7):length(handle);
  % These are the side-results of feedback_basket_init:
  basket = other_args{1};
  moveIdx = other_args{2};
  
  cursor= [0 0];
  cursor(moveIdx)= -1;
  counter= [0 0];
  moveStep= 2/(opt.trial_duration/opt.ms);
  if opt.balanced_sequence & opt.matchpoints<inf
    nn = floor(opt.matchpoints/opt.targets);
    sequence = ones(nn,1)*(1:opt.targets);
    sequence = [sequence(:);ceil(rand(opt.matchpoints-nn*opt.targets,1)*opt.targets)];
    sequence = sequence(randperm(opt.matchpoints));
    sequence = [ceil(rand(opt.free_balls,1)*opt.targets);sequence];
  end
  opt.reset = 0;
  waiting= 0;
  paused= 0;
  state= 0;
  if opt.free_balls>0
    do_set(h_counter,'Visible','off');
    free_balls=opt.free_balls;
  else
    free_balls=-1;
  end
  next_target=[];
  target = [];
end


if free_balls==0
  free_balls = -1;
  if strcmp(opt.show_counter,'on')
    do_set(h_counter,'Visible','on');
  end
end

if opt.changed==1 & opt.log==1,
  opt.changed= 0;
  % write log?
end

if strcmp(opt.status, 'pause') & ~paused,
  do_set(h_text, 'string','pause', 'visible','on');
  do_set(211);
  paused= 1;
end

if paused,
  if strcmp(opt.status, 'play'),
    do_set(209);
    waiting= 0;
    paused= 0;
    state= 0;
    if sum(counter)==opt.matchpoints % restart
      counter(:)= 0;
    end
  else
    do_set('+');
    return;
  end
end

switch(state),
 case 0,  %% countdown to start
  waiting= waiting + opt.ms;
  if waiting>=opt.countdown,
    do_set(h_text, 'visible','off');
    do_set(210);
    running_time= 0;
    state= 1;
  else
    str= sprintf('start in %.0f s', ceil((opt.countdown-waiting)/1000));
    do_set(h_text, 'string',str, 'visible','on');
  end
 case 1,  %% choose target
   if opt.balanced_sequence
     if ~isempty(target)
       do_set(h_rect2(target),'faceColor',opt.nontarget_color);
     end
     if ~isempty(next_target)
       target = next_target;
       do_set(h_rect2(next_target),'faceColor',opt.nontarget_color);
       if isempty(sequence)
         next_target = [];
       else
         next_target = sequence(1);
         sequence = sequence(2:end);
       end
     else
       target = sequence(1);
       next_target = sequence(2);
       sequence = sequence(3:end);
     end
     if ~isempty(next_target)
       do_set(h_rect2(next_target),'FaceColor',opt.target_color);
     end
   else
     target= ceil(opt.targets*rand);
   end
  do_set(target);
  do_set(h_rect(target), 'faceColor',opt.target_color);
  waiting= opt.time_before_free;
  state= 5;
 case 2,  %% wait for target hit
  cursor(moveIdx)= cursor(moveIdx) + moveStep;
  touch= pointinrect(cursor, basket);
  if ~isempty(touch),
    ishit= (touch==target);
    if free_balls>0
      free_balls=free_balls-1;
    else
      counter(2-ishit)= counter(2-ishit)+1;
      do_set(h_counter, 'string', sprintf('%d:%d', counter));
    end
    do_set(10+ishit);  %% 10:hit, 11:miss
    if ishit,
      do_set(h_rect(touch), 'faceColor',opt.hit_color);
    else
      do_set(h_rect(touch), 'faceColor',opt.miss_color);
    end
    if sum(counter)==opt.matchpoints,  %% game over
      bpd= bitrate(counter(1)/sum(counter), opt.targets);
      bpm= bpd*sum(counter)/running_time*60000;
      str= sprintf('%.1f bits/min', bpm);
      do_set(h_text, 'string',str, 'visible','on');
      counter(:)= 0;
      opt.status= 'pause';
      paused= 1;
      do_set(211);
    else
      do_set(setdiff(h_rect, h_rect(touch)), ...
          'faceColor',opt.nontarget_color);
      waiting= opt.time_after_hit;
      state= 3;
    end
  end
 case 3,  %% wait after target hit
  waiting= waiting - opt.ms;
  if waiting<0, 
    do_set(h_rect, 'faceColor',opt.nontarget_color);
    cursor(moveIdx)= -1;
    if opt.relative>0,
      cursor(3-moveIdx)= 0;
    end
    waiting= opt.time_before_next;
    if waiting>0,
      state= 4;
    else
      state= 1;
    end
  end
 case 4,  %% wait before choosing next target
  waiting= waiting - opt.ms;
  if waiting<0,
    state= 1;
  end
 case 5,  %% wait before freeing the cursor
  waiting= waiting - opt.ms;
  if waiting<0,
    state= 2;
  end
end

if state>0,
  running_time= running_time + opt.ms*(1+lost_packages);
end

if state==1 | state==2 | ...
    (~opt.relative & (state==3 | state==4)),
  if opt.relative==1,
    cursor(3-moveIdx)= cursor(3-moveIdx) + control/opt.damping;
  elseif opt.relative>0
    cursor(3-moveIdx)= opt.relative*cursor(3-moveIdx) + control*(1-opt.relative)/(1-opt.relative^opt.damping);    
  else
    cursor(3-moveIdx)= control;
  end
  cursor(3-moveIdx)= max(-1, min(1, cursor(3-moveIdx)));
end

cc= cursor;
if opt.cone>0,
  cf= min(1, (cursor(moveIdx)+1)/2/opt.cone);
  cc(3-moveIdx)= cf * cursor(3-moveIdx);
end

if state==2
  do_set(h_cursor, 'xData',cc(1), 'yData',cc(2));
end

do_set('+');
