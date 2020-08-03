function opt = feedback_cursor_1d_fixed_dur(fig, opt, ctrl);
%FEEDBACK_CURSOR_1D - BBCI Feedback: 1D Cursor Movement
%
%Synopsis:
% OPT= feedback_cursor_1d(FIG, OPT, CTRL)
%
%Arguments:
% FIG  - handle of figure
% OPT  - struct of optional properties, see below
% CTRL - control signal to be received from the BBCI classifier
%
%Output:
% OPT - updated structure of properties
%
%Optional Properties:
%  duration_show_selected: time to show selection 
%  duration_before_free: time after target presentation, before cursor
%    starts moving
%  time_per_trial: maximum duration of a trial
%  timeout_policy: 'miss', 'lastposition'
%  rate_control: switch for rate control (opposed to position control) 
%  speed: speed (in rate control) 1 means CTRL=1 moves in 1s from
%     center to target
%  cursor_on: switch to show (or hide) cursor
%  response_at: switch to show response (hit vs miss) (1) at 'center' area,
%     (2) at 'target' position, or (3) in the 'cursor' (cross).
%  trials_per_run: number of trials in one run 
%  free_trials: number of free trials (without counting hit or miss)
%  marker_active_spec: marker specification (cell array) of active cursor
%  marker_inactive_spec: marker specification (cell array) of inactive cursor
%  fixation_spec: marker specification (cell array) of fixation cross
%  background: color of figure background
%  color_hit: color for hit
%  color_miss: color for miss
%  color_target: color to indicate targets
%  color_nontarget: color of non-target
%  center_size: size of center area (for response in center)
%  target_width: width of target areas (normalized)
%  target_dist: distance of targets vertically from top and bottom
%  next_target: switch to show next target
%  next_target_width: width of next target indicator (normalized
%     within target width)
%  msg_spec: text specification (cell array) of messages 
%  points_spec: text specification (cell array) of points 
%  countdown: length of countdown before application starts [ms]
%  position: position of the figure (pixel)
%
%Markers written to parallel port
%    1: target on left side
%    2: target on right side
%   11: trial ended with cursor correctly on the left side
%   12: trial ended with cursor correctly on the right side
%   21: trial ended with cursor erroneously on the left side
%   22: trial ended with cursor erroneously on the right side
%   30: countdown starts
%   31: select goal side and indicate it
%   32: wait before cursor movement starts
%   33: in position control, cursor becomes active in center area
%   34: move cursor until target was hit or time-out
%   35: wait before next trial starts (or game is over)
%   41: first touch with cursor correctly on the left side
%   42: first touch with cursor correctly on the right side
%   51: first touch with cursor erroneously on the left side
%   52: first touch with cursor erroneously on the right side
%  200: init of the feedback
%  210: game status changed to 'play'
%  211: game status changed to 'pause'
%  212: game status changed to 'stop'
%
%See:
%  tester_cursor_1d_fixed_dur, feedback_cursor_1d_fixed_dur_init

% Author(s): Benjamin Blankertz, Nov-2006

persistent H HH state memo cfd

if ~isstruct(opt) | ~isfield(opt,'reset') 
  opt.reset = 1;
end

if opt.reset,
  opt.reset= 0;
  opt= set_defaults(opt, ...
        'countdown', 3000, ...
        'duration_show_selected', 1000, ...
        'duration_before_free', 750, ...
        'duration', 3500, ...
        'duration_jitter', 0, ...
        'duration_until_hit', 2000, ...
        'duration_break', 8000, ...
        'duration_break_fadeout', 1000, ...
        'duration_break_post_fadeout', 1000, ...
        'break_every', 0, ...
        'timeout_policy', 'lastposition', ...
        'rate_control', 1, ...
        'cursor_on', 1, ...
        'response_at', 'center', ...
        'trials_per_run',100,...
        'free_trials', 2, ...
        'cursor_active_spec', ...
          {'Color','k', 'Marker','+', 'MarkerSize',75, 'LineWidth', 15}, ...
        'cursor_inactive_spec', ...
          {'Color','k', 'Marker','.', 'MarkerSize',50, 'LineWidth', 1}, ...
        'fixation_spec', ...
          {'Marker','+', 'MarkerSize',50, 'LineWidth', 5}, ...
        'background', 0.9*[1 1 1], ...
        'color_hit', [0 0.8 0], ...
        'color_miss', [1 0 0], ...
        'color_target', [0 0 1], ...
        'color_nontarget', 0.7*[1 1 1], ...
        'color_center', 0.7*[1 1 1], ...
        'center_size', 0.3, ...
        'target_width', 0.1, ...
        'target_dist', 0.1, ...
        'next_target',0,...
        'next_target_width',0.2,...
        'damping_in_target', 'quadratic', ...
        'punchline', 1, ...
        'punchline_spec', {'Color',[0 0 0], 'LineWidth',3}, ...
        'punchline_beaten_spec', {'Color',[1 1 0], 'LineWidth',5}, ...
		    'msg_spec', {'FontSize',0.15}, ...
		    'points_spec', {'FontSize',0.075}, ...
        'parPort', 1,...
        'changed', 0,...
        'show_score', 1, ...
        'show_bit',0,...
        'log_state_changes',0, ...
        'log',1,...
        'fs', 25, ...
        'status', 'pause', ...
        'position', get(fig,'position'));
  
  [HH, cfd]= feedback_cursor_1d_fixed_dur_init(fig, opt);
  [handles, H]= fb_handleStruct2Vector(HH);
  
  do_set('init', handles, 'cursor_1d_fixed_dur', opt);
  do_set(200);

  seq= round(linspace(1, 2, opt.trials_per_run));
  pp= randperm(opt.trials_per_run);
  memo.sequence= [seq(pp), 0];
  memo.trial= -opt.free_trials;
  memo.stopwatch= 0;
  memo.x= 0;
  memo.lastdigit= NaN;
  memo.laststate= NaN;
  memo.laststatus= 'urknall';
  memo.timer= 0;
  memo.punch= [-1 1];
  state= -1;
end

if opt.changed,
  if ~strcmp(opt.status,memo.laststatus),
    memo.laststatus= opt.status;
    switch(opt.status),
     case 'play',
      do_set(210);
      if opt.rate_control & opt.cursor_on,
        do_set(H.cursor, 'Visible','off');
      end
      do_set(H.fixation, 'Visible','off');
      do_set(H.msg_punch, 'Visible','off');
      state= 0;
     case 'pause',
      do_set(211);
      do_set(H.msg, 'String','pause', 'Visible','on');
      state= -1;
     case 'stop';
      do_set(212);
      do_set(H.msg, 'String','stopped', 'Visible','on');
      state= -2;
    end
  end
end

if opt.changed==1,
  opt.changed= 0;
  do_set(H.msg, opt.msg_spec{:});
  do_set([H.points H.msg_punch], opt.points_spec{:});
  if opt.rate_control & ~strcmpi(opt.timeout_policy,'hitiflateral'),
    do_set(H.center, 'Visible','off');
  else
    do_set(H.center, 'FaceColor',opt.color_center, 'Visible','on');
  end
  if opt.cursor_on,
    do_set(H.cursor, 'Visible','on');
    do_set(H.fixation, 'Visible','off');
  else
    do_set(H.cursor, 'Visible','off');
  end
  if opt.show_score,
    do_set(H.points, 'Visible','on');
  else
    do_set(H.points, 'Visible','off');
  end
  if opt.punchline,
    do_set(H.punchline, 'Visible','on'),
  else
    do_set(H.punchline, 'Visible','off'),
  end
end

if state~=memo.laststate,
  memo.laststate= state;
  if state>=0 & opt.log_state_changes,
    do_set(30+state);
  end
  fprintf('state: %d\n', state);
end

if ~opt.rate_control & state>-2,
  memo.x= ctrl;
  memo.x= max(-1, min(1, memo.x));
end

if state==0,
  if memo.timer>=opt.countdown,
    do_set(H.msg, 'Visible','off');
    if opt.cursor_on,
      do_set(H.cursor, 'Visible','on');
    else
      do_set(H.fixation, 'Visible','on');
    end
    memo.touched_once= 0;
    state= 1;
  else
    memo.timer= memo.timer + 1000/opt.fs;
  end
  digit= ceil((opt.countdown-memo.timer)/1000);
  if digit~=memo.lastdigit,
    do_set(H.msg, 'String',int2str(digit), 'Visible','on');
    memo.lastdigit= digit;
  end
end

if state==1,  %% select goal side and indicate it
  memo.trial_duration= opt.duration + opt.duration_jitter*(rand*2-1);
  memo.trial= memo.trial + 1;
  if memo.trial==1,
    memo.stopwatch= 0;
  end    
  if memo.trial<1,
    memo.goal= ceil(2*rand);
    if memo.trial==0,
      memo.nextgoal= memo.sequence(memo.trial+1);
    else
      memo.nextgoal= 0;
    end
  else
    memo.goal= memo.sequence(memo.trial);
    memo.nextgoal= memo.sequence(memo.trial+1);
  end
  do_set(H.target(memo.goal), 'FaceColor',opt.color_target);
  if memo.nextgoal>0,
    do_set(H.next_target(memo.nextgoal), 'FaceColor',opt.color_target);
  end
  do_set(H.cursor, opt.cursor_inactive_spec{:});
  if opt.cursor_on,
    do_set(H.cursor', 'Visible','on');
  end
  memo.x= 0;
  memo.timer= 0;
  state= 2;
  do_set(memo.goal);
end

if state==2,  %% wait before cursor movement starts
  if memo.timer>opt.duration_before_free,
    state= 3;
  else
    memo.timer= memo.timer + 1000/opt.fs;
  end
end

if state==3,  %% in position control, cursor becomes active in center area
  if opt.rate_control | abs(memo.x)<opt.center_size,
    do_set(60);
    do_set(H.cursor, opt.cursor_active_spec{:});
    memo.trialwatch= 0;
    state= 4;
  end
end

if state==4,  %% move cursor until target was hit or time-out
  if opt.rate_control,
    if abs(memo.x)<1,
      memo.x= memo.x + ctrl/opt.fs/opt.duration_until_hit*1000;
    else
      switch(opt.damping_in_target),
       case 'linear',
        sec= (memo.trial_duration-opt.duration_until_hit)/1000;
        frames= 1 + ceil( opt.fs * sec );
        memo.x= memo.x + ctrl * cfd.target_width / frames;
       case 'quadratic',
        slope= 1000/opt.duration_until_hit;
        yy= 1 - (abs(memo.x)-1)/cfd.target_width;
        fac= yy*slope;
        memo.x= memo.x + ctrl*fac/opt.fs;
       otherwise
        error('option for damping_in_target unknown');
      end
    end
%    memo.x= max(-1-cfd.target_width, min(1+cfd.target_width, memo.x));
  end
  if abs(memo.x) >= 1,
    if ~memo.touched_once,
      memo.touched_once= 1;
      memo.preselected= sign(memo.x)/2 + 1.5;
      ishit= (memo.preselected==memo.goal);
      do_set(30+10*(2-ishit)+memo.preselected);
    end
  end
  if memo.trialwatch>=memo.trial_duration,
    switch(lower(opt.timeout_policy)),
     case 'miss',
      memo.selected= [];
      ishit= 0;
      do_set(23);
     case 'lastposition',
      memo.selected= sign(memo.x)/2 + 1.5;
      ishit= (memo.selected==memo.goal);
      do_set(10*(2-ishit)+memo.selected);
     case 'hitiflateral',
      if abs(memo.x)>opt.center_size,
        %% cursor is lateral enough (outside center): count as hit
        memo.selected= sign(memo.x)/2 + 1.5;
        ishit= (memo.selected==memo.goal);
        do_set(10*(2-ishit)+memo.selected);
      else
        %% cursor is within center area: count as miss
        memo.selected= [];
        ishit= -1;
        memo.rejected= memo.rejected + 1;
        do_set(25);  %% reject
      end
     otherwise
      error('unknown value for OPT.timeout_policy');
    end
    if memo.trial>0,
      memo.ishit(memo.trial)= (ishit==1);
      nHits= sum(memo.ishit(1:memo.trial));
      nMisses= memo.trial - memo.rejected - nHits;
      do_set(H.points(1), 'String',['HIT: ' int2str(nHits)]);
      do_set(H.points(2), 'String',['MISS: ' int2str(nMisses)]);
      if ishit==1,
        ii= memo.selected;
        if abs(memo.x) > abs(memo.punch(ii)),  %% beaten the punchline?
          memo.punch(ii)= memo.x;
          do_set(H.punchline(ii), 'XData',[1 1]*memo.x, ...
                 opt.punchline_beaten_spec{:});
        end
      end
    end
    switch(lower(opt.response_at)),
     case 'center',
      memo.H_indicator= H.center;
      ind_prop= 'FaceColor';
      do_set(H.center, 'Visible','on');
     case 'target',
      if isempty(memo.selected), %% timeout trial
                                 %% (with opt.timeout_policy='miss')
        memo.H_indicator= H.cursor;
        ind_prop= 'Color';
      else
        memo.H_indicator= H.target(memo.selected);
        ind_prop= 'FaceColor';
      end
     case 'cursor',
      memo.H_indicator= H.cursor;
      ind_prop= 'Color';
     otherwise,
      warning('value for property ''response at'' unrecognized.');
      memo.H_indicator= H.target(memo.selected);
    end
    switch(ishit),
     case 1,
      do_set(memo.H_indicator, ind_prop,opt.color_hit);
     case 0,
      do_set(memo.H_indicator, ind_prop,opt.color_miss);
     case -1,
      do_set(memo.H_indicator, ind_prop,opt.color_reject);
    end
    memo.timer= 0;
    state= 5;
  end
  memo.trialwatch= memo.trialwatch + 1000/opt.fs;
end

if state==5,  %% wait before next trial starts (or game is over)
  if memo.timer>opt.duration_show_selected,
    switch(lower(opt.response_at)),
     case 'center',
      do_set(H.center, 'Visible','off');
     case 'target',
      do_set(H.center, 'FaceColor',opt.color_center);
     case 'cursor',
    end
    do_set(H.target, 'FaceColor',opt.color_nontarget);
    do_set(H.next_target, 'FaceColor',opt.color_nontarget);
    do_set(H.punchline, opt.punchline_spec{:});
    if memo.trial==opt.trials_per_run,
      if ~opt.show_score,  %% show score at least at the end
        do_set(H.points, 'Visible','on');
      end
      if opt.show_bit,
        minutes= memo.stopwatch/1000/60;
        bpm= bitrate(mean(memo.ishit)) * opt.trials_per_run / minutes;
        msg= sprintf('%.1f bits/min', bpm);
        do_set(H.msg, 'String',msg, 'Visible','on');
      end
      msg= sprintf('punch at [%d %d]', ...
                   round(100*(memo.punch-sign(memo.punch))/cfd.target_width));
      do_set(H.msg_punch, 'String',msg, 'Visible','on');
      do_set(H.fixation, 'Visible','off');
      if opt.rate_control,
        do_set(H.cursor, 'Visible','off');
      else
        do_set(H.cursor, opt.cursor_inactive_spec{:});
      end
      state= -1;
    elseif opt.break_every>0 & mod(memo.trial,opt.break_every)==0,
      nHits= sum(memo.ishit(1:memo.trial));
      msg= sprintf('%d : %d', nHits, memo.trial-nHits);
      do_set(H.msg, 'String',msg, 'Visible','on');
      memo.timer= 0;
      state= 6;
    else
      state= 1;
    end
  else
    memo.timer= memo.timer + 1000/opt.fs;
  end
end

if state==6,   %% give a break where the score is shown
  if memo.timer>opt.duration_break,
    do_set(H.cursor, 'Visible','off');
    memo.timer= 0;
    state= 7;
  end
  memo.timer= memo.timer + 1000/opt.fs;
end

if state==7,   %% score fade-out at the end of the break
  if memo.timer>opt.duration_break_fadeout+opt.duration_break_post_fadeout,
    do_set(H.msg, 'Visible','off', 'Color',[0 0 0]);
    state= 1;
  elseif memo.timer<=opt.duration_break_fadeout,
    fade= (opt.duration_break_fadeout-memo.timer)/opt.duration_break_fadeout;
    do_set(H.msg, 'Color',[0 0 0]*fade + opt.background*(1-fade));
  end
  memo.timer= memo.timer + 1000/opt.fs;
end


memo.stopwatch= memo.stopwatch + 1000/opt.fs;

do_set(H.cursor, 'XData',memo.x);
do_set('+');
