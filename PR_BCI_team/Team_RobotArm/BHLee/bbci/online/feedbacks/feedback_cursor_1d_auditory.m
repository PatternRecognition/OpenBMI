function opt = feedback_cursor_1d_auditory(fig, opt, ctrl);
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
%  fixed_trial_duration: if true, the trial duration is fixed to time_per_trial
%  timeout_policy: 'miss', 'lastposition'
%  rate_control: switch for rate control (opposed to position control) 
%  speed: speed (in rate control) 1 means CTRL=1 moves in 1s from
%     center to target
%  cursor_on: switch to show (or hide) cursor
%  response_in_center: switch to show response (hit vs miss) in center
%     (or at target) position
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
%   11: target was correctly hit on the left side
%   12: target was correctly hit on the right side
%   21: cursor erroneously hit the left side while target was right
%   22: cursor erroneously hit the right side while target was left
%   23: trial ended by time-out and opt.timeout_policy='miss'
%   30: countdown starts
%   31: select goal side and indicate it
%   32: wait before cursor movement starts
%   33: in position control, cursor becomes active in center area
%   34: move cursor until target was hit or time-out
%   35: wait before next trial starts (or game is over)
%   36: state switch
%  200: init of the feedback
%  210: game status changed to 'play'
%  211: game status changed to 'pause'
%  212: game status changed to 'stop'
%
%See:
%  tester_cursor_1d_auditory, feedback_cursor_1d_auditory_init

% Author(s): Benjamin Blankertz, Nov-2006

global SOUND_DIR
persistent H HH state memo

if ~isstruct(opt) | ~isfield(opt,'reset') 
  opt.reset = 1;
end

if opt.reset,
  opt.reset= 0;
  opt= set_defaults(opt, ...
        'countdown', 3000, ...
        'duration_show_selected', 1000, ...
        'duration_before_free', 750, ...
        'duration_after_switch', 5000, ...
        'fixed_trial_duration', 0, ...
        'time_per_trial', 5000, ...
        'timeout_policy', 'lastposition', ...
        'rate_control', 1, ...
        'speed', 1, ...
        'cursor_on', 0, ...
        'response_in_center', 1, ...
        'trials_per_run',100,...
        'free_trials', 2, ...
        'cursor_active_spec', ...
          {'Marker','+', 'MarkerSize',75, 'LineWidth', 15}, ...
        'cursor_inactive_spec', ...
          {'Marker','.', 'MarkerSize',50, 'LineWidth', 1}, ...
        'fixation_spec', ...
          {'Marker','+', 'MarkerSize',50, 'LineWidth', 5}, ...
        'background', 0.9*[1 1 1], ...
        'color_hit', [0 0.8 0], ...
        'color_miss', [1 0 0], ...
        'color_target', [0 0 1], ...
        'color_nontarget', 0.7*[1 1 1], ...
        'color_center', 0.7*[1 1 1], ...
        'center_size', 0.3, ...
        'target_width', 0.075, ...
        'target_dist', 0.1, ...
        'next_target',0,...
        'next_target_width',0.2,...
        'switch_states', 1, ...
        'switch_block_duration', 45*1000, ...
        'sounds_targets', {'links','rechts'}, ...
        'sounds_hitmiss', {'sound_miss', 'sound_hit'}, ...
        'sounds_counting', {'eins','zwei','drei','vier','fuenf'}, ...
        'sounds_switch', {'Augen_offen','Augen_zu'}, ...
		    'msg_spec', {'FontSize',0.15}, ...
		    'points_spec', {'FontSize',0.075}, ...
        'parPort', 1,...
        'changed',0,...
        'show_bit',0,...
        'log',1,...
        'fs', 25, ...
        'status', 'pause', ...
        'position', get(fig,'position'));
  
  HH= feedback_cursor_1d_auditory_init(fig, opt);
  [handles, H]= fb_handleStruct2Vector(HH);
  
  do_set('init', handles, 'cursor_1d_auditory', opt);
  do_set(200);

  for ii= 1:2,
    [memo.sound(ii).wav, memo.sound(ii).fs]= ...
        wavread([SOUND_DIR opt.sounds_targets{ii} '.wav']);
    [memo.sound(2+ii).wav, memo.sound(2+ii).fs]= ...
        wavread([SOUND_DIR opt.sounds_hitmiss{ii} '.wav']);
  end
  for ii= 1:length(opt.sounds_counting),
    [memo.sound(4+ii).wav, memo.sound(4+ii).fs]= ...
        wavread([SOUND_DIR opt.sounds_counting{ii} '.wav']);
  end
  memo.iss= 4+ii;
  for ii= 1:length(opt.sounds_switch),
    [memo.sound(memo.iss+ii).wav, memo.sound(memo.iss+ii).fs]= ...
        wavread([SOUND_DIR opt.sounds_switch{ii} '.wav']);
  end
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
      memo.timer= 0;
      if opt.switch_states,
        memo.switchstate= 1;
        wavplay(memo.sound(memo.iss+memo.switchstate).wav, ...
                memo.sound(memo.iss+memo.switchstate).fs, 'async');
        state= 6;
      else
        state= 0;
      end
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
  do_set(H.points, opt.points_spec{:});
  if opt.rate_control,
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
end

if state~=memo.laststate,
  memo.laststate= state;
  if state>=0,
    do_set(30+state);
  end
  fprintf('state: %d\n', state);
end

if ~opt.rate_control & state>-2,
  memo.x= ctrl;
  memo.x= max(-1, min(1, memo.x));
end

if state==0,  %% do the count-down
  if memo.timer>=opt.countdown,
    do_set(H.msg, 'Visible','off');
    if opt.cursor_on,
      do_set(H.cursor, 'Visible','on');
    else
      do_set(H.fixation, 'Visible','on');
    end
    memo.last_switch_time= memo.stopwatch;
    state= 1;
  else
    memo.timer= memo.timer + 1000/opt.fs;
  end
  digit= ceil((opt.countdown-memo.timer)/1000);
  if digit~=memo.lastdigit,
    if digit<=length(opt.sounds_counting) & digit>0,
      wavplay(memo.sound(4+digit).wav, memo.sound(4+digit).fs, 'async');
    end
    do_set(H.msg, 'String',int2str(digit), 'Visible','on');
    memo.lastdigit= digit;
  end
end

if state==1,  %% select goal side and indicate it
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
  wavplay(memo.sound(memo.goal).wav, memo.sound(memo.goal).fs, 'async');
  do_set(H.target(memo.goal), 'FaceColor',opt.color_target);
  if memo.nextgoal>0,
    do_set(H.next_target(memo.nextgoal), 'FaceColor',opt.color_target);
  end
  do_set(H.cursor, opt.cursor_inactive_spec{:});
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
    memo.x= memo.x + ctrl*opt.speed/opt.fs;
    memo.x= max(-1, min(1, memo.x));
  end
  if (abs(memo.x) > 1-2*opt.target_width & ~opt.fixed_trial_duration) ...
        | memo.trialwatch>opt.time_per_trial,
    if memo.trialwatch>opt.time_per_trial & strcmpi(opt.timeout_policy, 'miss'),
      memo.selected= [];
      ishit= 0;
      do_set(23);
    else
      memo.selected= sign(memo.x)/2 + 1.5;
      ishit= memo.selected==memo.goal;
      do_set(10*(2-ishit)+memo.selected);
    end
    wavplay(memo.sound(3+ishit).wav, memo.sound(3+ishit).fs, 'async');
    if memo.trial>0,
      memo.ishit(memo.trial)= ishit;
      nHits= sum(memo.ishit(1:memo.trial));
      do_set(H.points(1), 'String',['HIT: ' int2str(nHits)]);
      do_set(H.points(2), 'String',['MISS: ' int2str(memo.trial-nHits)]);
    end
    if isempty(memo.selected),     %% in timeout trials (with opt.timeout_policy=='miss') display red center
      memo.H_indicator= H.center;
      do_set(H.center, 'Visible','on');
    else
      if opt.response_in_center,
        memo.H_indicator= H.center;
        do_set(H.center, 'Visible','on');
      else
        memo.H_indicator= H.target(memo.selected);
      end
    end
    if ishit,
      do_set(memo.H_indicator, 'FaceColor',opt.color_hit);
    else
      do_set(memo.H_indicator, 'FaceColor',opt.color_miss);
    end
    memo.timer= 0;
    state= 5;
  end
  memo.trialwatch= memo.trialwatch + 1000/opt.fs;
end

if state==5,  %% wait before next trial starts (or game is over)
  if memo.timer>opt.duration_show_selected,
    if opt.response_in_center,
      if opt.rate_control,
        do_set(H.center, 'Visible','off');
      else
        do_set(H.center, 'FaceColor',opt.color_center);
      end
    end
    do_set(H.target, 'FaceColor',opt.color_nontarget);
    do_set(H.next_target, 'FaceColor',opt.color_nontarget);
    if memo.trial==opt.trials_per_run,
      minutes= memo.stopwatch/1000/60;
      bpm= bitrate(mean(memo.ishit)) * opt.trials_per_run / minutes;
      msg= sprintf('%.1f bits/min', bpm);
      do_set(H.msg, 'String',msg, 'Visible','on');
      do_set(H.fixation, 'Visible','off');
      if opt.rate_control,
        do_set(H.cursor, 'Visible','off');
      else
        do_set(H.cursor, opt.cursor_inactive_spec{:});
      end
      state= -1;
    else
      [(memo.stopwatch-memo.last_switch_time) opt.switch_block_duration]
      if opt.switch_states & ...
            ((memo.stopwatch-memo.last_switch_time) ...
             >= opt.switch_block_duration),
        memo.switchstate= mod(memo.switchstate, length(opt.sounds_switch))+1;
        wavplay(memo.sound(memo.iss+memo.switchstate).wav, ...
                memo.sound(memo.iss+memo.switchstate).fs, 'async');
        memo.timer= 0;
        state= 6;
      else
        state= 1;
      end
    end
  else
    memo.timer= memo.timer + 1000/opt.fs;
  end
end

if state==6,  %% wait after announcement of new state
  if memo.timer>opt.duration_after_switch,
    memo.timer= 0;
    state= 0;
  else
    memo.timer= memo.timer + 1000/opt.fs;
  end  
end

memo.stopwatch= memo.stopwatch + 1000/opt.fs;

do_set(H.cursor, 'XData',memo.x);
do_set('+');
