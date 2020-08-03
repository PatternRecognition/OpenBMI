function opt = feedback_cursor_1d_pro(fig, opt, ctrl);
%FEEDBACK_CURSOR_1D_PRO - BBCI Feedback: 1D Cursor Movement
%
%Synopsis:
% OPT= feedback_cursor_1d_pro(FIG, OPT, CTRL)
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
%  duration_show_selected_jitter: jitter on duration_show_selected.
%  duration_isi: inter-stimulus interval.
%  duration_isi_jitter: jitter on duration_isi.
%  duration_before_free: time after target presentation, before cursor
%    starts moving
%  duration: (maximum) duration of a trial
%  timeout_policy: 'miss', 'reject', 'lastposition', 'hitiflateral'
%  rate_control: switch for rate control (opposed to position control) 
%  speed: speed (in rate control) 1 means CTRL=1 moves in 1s from
%     center to target
%  cursor_on: switch to show (or hide) cursor
%  response_at: switch to show response (hit vs miss) (1) at 'center' area,
%     or (2) at 'target' position, or (3) in the 'cursor' (cross), or
%     (4) 'none' not at all.
%  trials_per_run: number of trials in one run 
%  free_trials: number of free trials (without counting hit or miss)
%  break_every: number of trial after which a break is inserted, 0 means no
%     breaks. Default: 0.
%  break_show_score: 1 means during breaks the score is shown, 0 means not.
%     Default: 1.
%  marker_active_spec: marker specification (cell array) of active cursor
%  marker_inactive_spec: marker specification (cell array) of inactive cursor
%  fixation_spec: marker specification (cell array) of fixation cross
%  background: color of figure background
%  color_hit: color for hit
%  color_miss: color for miss
%  color_reject: color for reject
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
%   23: trial ended by time-out and opt.timeout_policy='miss'
%   24: trial ended by time-out and opt.timeout_policy='reject'
%   25: trial ended by time-out and opt.timeout_policy='hitiflateral'
%          and cursor was not lateral enough: reject
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
%   60: cursor becomes active.
%   70: start of adaptation phase 1 (rotating cursor)
%   71: end of adaptation phase 1
%  100+x: Logfile with number x has been opened for writing.
%  200: init of the feedback
%  210: game status changed to 'play'
%  211: game status changed to 'pause'
%  212: game status changed to 'stop'
%  213: game ends
%
%See:
%  tester_cursor_1d_pro, feedback_cursor_1d_pro_init

% Author(s): Benjamin Blankertz, Nov-2006

persistent H HH state memo cfd

if ~isstruct(opt) | ~isfield(opt,'reset') 
  opt.reset = 1;
end

if opt.reset,
  opt.reset= 0;
  [opt, isdefault]= ...
      set_defaults(opt, ...
        'countdown', 3000, ...
        'duration', 3500, ...
        'duration_show_selected', 1000, ...
        'duration_show_selected_jitter', 0, ...
        'duration_before_free', 750, ...
        'duration_jitter', 0, ...
	'duration_isi',0,...
	'duration_isi_jitter',0,...	   
        'duration_until_hit', 2000, ...
        'duration_break', 8000, ...
        'duration_break_fadeout', 1000, ...
        'duration_break_post_fadeout', 1000, ...
        'touch_terminates_trial', 1, ...
        'break_every', 0, ...
        'break_show_score', 1, ...
        'break_endswith_countdown', 0, ...
        'timeout_policy', 'hitiflateral', ...
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
        'color_reject', [0.8 0 0.8], ...
        'color_target', [0 0 1], ...
        'color_nontarget', 0.7*[1 1 1], ...
        'color_center', 0.7*[1 1 1], ...
        'center_size', 0.3, ...
        'target_width', 0.1, ...
        'target_dist', 0.1, ...
        'next_target',0,...
        'next_target_width',0.2,...
        'damping_in_target', 'quadratic', ...
        'punchline', 0, ...
        'punchline_spec', {'Color',[0 0 0], 'LineWidth',3}, ...
        'punchline_beaten_spec', {'Color',[1 1 0], 'LineWidth',5}, ...
		    'msg_spec', {'FontSize',0.15}, ...
		    'points_spec', {'FontSize',0.075}, ...
        'adapt_trials', 0, ...
        'adapt_duration_rotation', 40*1000, ...
        'adapt_duration_msg', 2*1000, ...
        'adapt_radius', 0.5, ...
        'adapt_rotation_speed', 0.4, ...
        'parPort', 1,...
        'changed', 0,...
        'show_score', 1, ...
        'show_rejected', 1, ...
        'show_bit', 1,...
        'log_state_changes',0, ...
        'log',1,...
        'fs', 25, ...
        'status', 'pause', ...
        'position', get(fig,'position'));
  
  if ~opt.touch_terminates_trial & isdefault.punchline,
    [opt, isdefault]= opt_overrideIfDefault(opt, isdefault, ...
                                            'punchline', 1);
  end
  if ~ismember(opt.timeout_policy,{'hitiflateral','reject'}) ...
        & isdefault.show_rejected,
    [opt, isdefault]= opt_overrideIfDefault(opt, isdefault, ...
                                            'show_rejected', 0);
  end
  if opt.adapt_trials>0 & length(opt.free_trials)==1,
    opt.free_trials= [0 opt.free_trials];
  end
  
  [HH, cfd]= feedback_cursor_1d_pro_init(fig, opt);
  [handles, H]= fb_handleStruct2Vector(HH);
  
  do_set('init', handles, 'cursor_1d_pro', opt);
  do_set(200);

  memo.stopwatch= 0;
  memo.x= 0;
  memo.degree= 0;
  memo.lastdigit= NaN;
  memo.laststate= NaN;
  memo.laststatus= 'urknall';
  memo.timer= 0;
  if opt.adapt_trials>0,
    memo.modus= 3;
  else
    memo.modus= 1;
  end
  state= -1;
end

if opt.changed,
  if ~strcmp(opt.status, memo.laststatus),
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
    memo.center_visible= 0;
    do_set(H.center, 'Visible','off');
  else
    memo.center_visible= 1;
    do_set(H.center, 'FaceColor',opt.color_center, 'Visible','on');
  end
  if opt.cursor_on,
    if state>0,
      do_set(H.cursor, 'Visible','on');
    end
    do_set(H.fixation, 'Visible','off');
  else
    do_set(H.cursor, 'Visible','off');
  end
  if opt.show_score,
    do_set(H.points, 'Visible','on');
  else
    do_set(H.points, 'Visible','off');
  end
  if opt.show_rejected,
    do_set(H.rejected_counter, 'Visible','on');
  else
    do_set(H.rejected_counter, 'Visible','off');
  end
  if opt.punchline,
    do_set(H.punchline, 'Visible','on'),
  else
    do_set(H.punchline, 'Visible','off'),
  end
  if ~opt.cursor_on & strcmp(opt.response_at, 'cursor'),
    opt.response_at= 'center';
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
  digit= ceil((opt.countdown-memo.timer)/1000);
  if digit~=memo.lastdigit,
    do_set(H.msg, 'String',int2str(digit), 'Visible','on');
    memo.lastdigit= digit;
  end
  if memo.timer>=opt.countdown,
    %% countdown terminates: prepare run
    do_set(H.msg, 'Visible','off');
    if memo.modus==0,      %% during break inbetween run
      state= 1;
      if opt.cursor_on,
        do_set(H.cursor, 'Visible','on');
      else
        do_set(H.fixation, 'Visible','on');
      end
    else
      memo.punch= [-1 1];
      memo.ishit= [];
      memo.rejected= 0;
      if memo.modus==3,  %% rotation adaptation coming up
        do_set(70);
        memo.timer= 0;
        state= 20;
      else
        if memo.modus==2,    %% bias adaptation coming up
          nTrials= 2*opt.adapt_trials;
        else                 %% start of regular feedback part
          nTrials= opt.trials_per_run;
        end
        seq= round(linspace(1, 2, nTrials));
        pp= randperm(nTrials);
        memo.sequence= [seq(pp), 0];
        memo.trial= -opt.free_trials(memo.modus);
        if opt.cursor_on,
          do_set(H.cursor, 'Visible','on');
        else
          do_set(H.fixation, 'Visible','on');
        end
        memo.touched_once= 0;
        state= 1;
      end
    end
  else
    memo.timer= memo.timer + 1000/opt.fs;
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
  if memo.timer>=opt.duration_before_free,
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
  
  trial_terminates= 0;
  
  %% cursor touches target
  if abs(memo.x) >= 1,
    if opt.touch_terminates_trial,
      trial_terminates= 1;
      memo.selected= sign(memo.x)/2 + 1.5;
      ishit= (memo.selected==memo.goal);
      do_set(10*(2-ishit)+memo.selected);
    elseif ~memo.touched_once,
      memo.touched_once= 1;
      memo.preselected= sign(memo.x)/2 + 1.5;
      ishit= (memo.preselected==memo.goal);
      do_set(30+10*(2-ishit)+memo.preselected);
    end
  end
  
  %% timeout
  if memo.trialwatch >= memo.trial_duration,
    trial_terminates= 1;
    switch(lower(opt.timeout_policy)),
     case 'miss',
      memo.selected= [];
      ishit= 0;
      do_set(23);  %% reject by timeout
     case 'reject',
      memo.selected= [];
      ishit= -1;
      do_set(24);  %% reject by timeout
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
        %% cursor is within center area: count as reject
        memo.selected= [];
        ishit= -1;
        do_set(25);  %% reject by timeout and position
      end
     otherwise
      error('unknown value for OPT.timeout_policy');
    end
  end
  
  %% trial terminates: show response and update score
  if trial_terminates,
    if memo.trial>0,
      if ishit==-1,
        memo.rejected= memo.rejected + 1;
      end      
      memo.ishit(memo.trial)= (ishit==1);
      nHits= sum(memo.ishit(1:memo.trial));
      nMisses= memo.trial - memo.rejected - nHits;
      do_set(H.points(1), 'String',['HIT: ' int2str(nHits)]);
      do_set(H.points(2), 'String',['MISS: ' int2str(nMisses)]);
      do_set(H.rejected_counter, 'String',['REJ: ' int2str(memo.rejected)]);
      if ishit==1 & ~opt.touch_terminates_trial,
        ii= memo.selected;
        if abs(memo.x) > abs(memo.punch(ii)),  %% beaten the punchline?
          memo.punch(ii)= memo.x;
          do_set(H.punchline(ii), 'XData',[1 1]*memo.x, ...
                 opt.punchline_beaten_spec{:});
        end
      end
    end
    reset_target_color= 0;
    switch(lower(opt.response_at)),
     case 'center',
      memo.H_indicator= H.center;
      ind_prop= 'FaceColor';
      do_set(H.center, 'Visible','on');
      reset_target_color= 1;
     case 'target',
      if isempty(memo.selected), %% timeout trial
                                 %% (with opt.timeout_policy='miss')
        memo.H_indicator= H.cursor;
        ind_prop= 'Color';
        reset_target_color= 1;
      else
        memo.H_indicator= H.target(memo.selected);
        ind_prop= 'FaceColor';
      end
     case 'cursor',
      memo.H_indicator= H.cursor;
      ind_prop= 'Color';
      reset_target_color= 1;
     case 'none',
      memo.H_indicator= [];
      reset_target_color= 1;
     otherwise,
      warning('value for property ''response at'' unrecognized.');
      memo.H_indicator= H.target(memo.selected);
    end
    if reset_target_color,
      do_set(H.target, 'FaceColor',opt.color_nontarget);
    end
    if ~isempty(memo.H_indicator),
      switch(ishit),
       case 1,
        do_set(memo.H_indicator, ind_prop,opt.color_hit);
       case 0,
        do_set(memo.H_indicator, ind_prop,opt.color_miss);
       case -1,
        do_set(memo.H_indicator, ind_prop,opt.color_reject);
      end
    end
    memo.timer= 0;
    state= 5;
  end
  memo.trialwatch= memo.trialwatch + 1000/opt.fs;
end

if state==5,  %% wait before next trial starts (or game is over)
  if memo.timer >= opt.duration_show_selected+opt.duration_show_selected_jitter*(2*rand-1),
    switch(lower(opt.response_at)),
     case 'center',
      if memo.center_visible,
        do_set(H.center, 'FaceColor',opt.color_center);
      else
        do_set(H.center, 'Visible','off');
      end
     case 'target',
     case 'cursor',
     case 'none',
    end
    do_set(H.target, 'FaceColor',opt.color_nontarget);
    do_set(H.next_target, 'FaceColor',opt.color_nontarget);
    do_set(H.punchline, opt.punchline_spec{:});
    if memo.trial==length(memo.sequence)-1,
      if memo.modus==2,
        memo.timer= 0;
        state= 21;
      else
        %% game over
        if ~opt.show_score,  %% show score at least at the end
          do_set(H.points, 'Visible','on');
        end
        if opt.show_bit,
          minutes= memo.stopwatch/1000/60;
          acc= sum(memo.ishit)/(memo.trial-memo.rejected);
          bpm= bitrate(acc) * opt.trials_per_run / minutes;
          msg= sprintf('%.1f bits/min', bpm);
          do_set(H.msg, 'String',msg, 'Visible','on');
        else
          msg= sprintf('- game over -');
          do_set(H.msg, 'String',msg, 'Visible','on');
        end
        if opt.punchline,
          msg= sprintf('punch at  [%d %d]', ...
                       round(100*(memo.punch-sign(memo.punch))/cfd.target_width));
          do_set(H.msg_punch, 'String',msg, 'Visible','on');
        end
        do_set(H.fixation, 'Visible','off');
        if opt.rate_control,
          do_set(H.cursor, 'Visible','off');
        else
          do_set(H.cursor, opt.cursor_inactive_spec{:});
        end
        do_set(213);
        state= -1;
      end
    elseif opt.break_every>0 & memo.trial>0 & mod(memo.trial,opt.break_every)==0,
      if opt.break_show_score,
        nHits= sum(memo.ishit(1:memo.trial));
        msg= sprintf('%d : %d', nHits, memo.trial - nHits - memo.rejected);
        do_set(H.msg, 'String',msg, 'Visible','on');
      end
      do_set(H.center, 'Visible','off');
      do_set(H.cursor, 'Visible','off');
      do_set(H.fixation, 'Visible','off');
      memo.timer= 0;
      state= 6;
    else
      state= 8;
      memo.timer = 0;
      return;
    end
  else
    memo.timer= memo.timer + 1000/opt.fs;
  end
end

if state==6,   %% give a break where the score is shown
  if memo.timer>opt.duration_break,
    memo.timer= 0;
    state= 7;
  end
  memo.timer= memo.timer + 1000/opt.fs;
end

if state==7,   %% score fade-out at the end of the break
  if memo.timer>opt.duration_break_fadeout+opt.duration_break_post_fadeout,
    if opt.break_endswith_countdown,
      memo.modus= 0;
      memo.timer= 0;
      state= 0;
    else
      state= 1;
    end
  elseif memo.timer<=opt.duration_break_fadeout,
    fade= (opt.duration_break_fadeout-memo.timer)/opt.duration_break_fadeout;
    do_set(H.msg, 'Color',[0 0 0]*fade + opt.background*(1-fade));
    memo.fadeout_finished= 0;
  else
    if ~memo.fadeout_finished,
      memo.fadeout_finished= 1;
      if memo.center_visible,
        do_set(H.center, 'Visible','on');
      end
      do_set(H.msg, 'Visible','off', 'Color',[0 0 0]);
      if ~opt.cursor_on,
        do_set(H.fixation, 'Visible','on');
      end
    end
  end
  memo.timer= memo.timer + 1000/opt.fs;
end

if state==8, %% inter-stimulus interval: everything inactive.
  if memo.timer>opt.duration_isi+opt.duration_isi_jitter,
    state = 1;
    opt = feedback_cursor_1d_pro(fig, opt, ctrl);
    return;
  end
  memo.timer= memo.timer + 1000/opt.fs;  
end

if state==20,  %% circular moving cross for adaption
  memo.degree= memo.degree + opt.adapt_rotation_speed*2*pi/opt.fs;
  memo.x = opt.adapt_radius*cos(memo.degree);
  memo.y = opt.adapt_radius*sin(memo.degree);
  do_set(H.cursor, 'XData',memo.x, 'YData',memo.y,'Visible','on');
  if memo.timer >= opt.adapt_duration_rotation,
    do_set(71);
    memo.timer= 0;
    state= 21;
  else
    memo.timer = memo.timer + 1000/opt.fs;
  end
end

if state==21,
  if memo.timer==0,
    do_set(H.cursor, 'Visible','off', 'XData',0, 'YData',0);
    do_set(H.msg, 'String','Adapting ...', 'Visible','on'); 
    do_set(H.points(1), 'String','HIT: 0');
    do_set(H.points(2), 'String','MISS: 0');
    do_set(H.rejected_counter, 'String','REJ: 0');
  end
  if memo.timer >= opt.adapt_duration_msg,
    memo.modus= memo.modus - 1;
    memo.timer= 0;
    memo.rejected= 0;
    state= 0;
  else
    memo.timer = memo.timer + 1000/opt.fs;
  end
end
 
memo.stopwatch= memo.stopwatch + 1000/opt.fs;

do_set(H.cursor, 'XData',memo.x);
do_set('+');
