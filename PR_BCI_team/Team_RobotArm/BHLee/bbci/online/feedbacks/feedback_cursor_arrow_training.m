function opt = feedback_cursor_arrow_training(fig, opt, ctrl1, ctrl2, ctrl3);
%FEEDBACK_CURSOR_ARROW - BBCI Feedback: Cursor Movement with Arrow Cues
%
%Synopsis:
% OPT= feedback_cursor_arrow(FIG, OPT, CTRL1, CTRL2, CTRL3)
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
%  timeout_policy: 'miss', 'reject', 'lastposition', 'hitiflateral'
%  proportional_control: 
%  cursor_on: switch to show (or hide) cursor
%  trials_per_run: number of trials in one run 
%  free_trials: number of free trials (without counting hit or miss)
%  break_every: number of trial after which a break is inserted, 0 means no
%     breaks. Default: 0.
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
%  countdown: length of countdown before application starts [ms]
%  position: position of the figure (pixel)
%
%Markers written to parallel port
%    1: target on left side
%    2: target on right side
%   11: trial ended with cursor correctly on the left side
%   12: trial ended with cursor correctly on the right side
%   29: end of trial 
%   30: countdown starts
%   31: select goal side and indicate it
%   32: wait before cursor movement starts
%   33: in position control, cursor becomes active in center area
%   34: move cursor until target was hit or time-out
%   35: wait before next trial starts (or game is over)
%  200: init of the feedback
%  210: game status changed to 'play'
%  211: game status changed to 'pause'
%  255: game ends
%
%See:
%  tester_cursor_1d_pro, feedback_cursor_1d_pro_init

% Author(s): Benjamin Blankertz, Nov-2006

persistent H HH state memo cfd

if ~isstruct(opt) | ~isfield(opt,'reset') 
  opt.reset = 1;
end

if opt.reset,
  global VP_SCREEN
  opt.reset= 0;
  [opt, isdefault]= ...
      set_defaults(opt, ...
        'countdown', 7000, ...
        'classes', {'left','right','foot'}, ...
        'trigger_classifier_list', {}, ...
        'fallback', 0.15, ...
        'gain_error', 0.2, ...
        'duration', 3000, ...
        'duration_jitter', 0, ...
        'duration_before_free', 1000, ...
        'duration_show_selected', 1000, ...
        'duration_blank', 2000, ...
        'duration_until_hit', 1500, ...
        'duration_break', 15000, ...
        'duration_break_fadeout', 1000, ...
        'duration_break_post_fadeout', 1000, ...
        'proportional_control', 0, ...
        'remove_cue_at_end_of_trial', 0, ...
        'break_every', 20, ...
        'break_show_score', 1, ...
        'break_endswith_countdown', 1, ...
        'timeout_policy', 'hitiflateral', ...
        'cursor_on', 1, ...
        'trials_per_run',100,...
        'cursor_active_spec', ...
                   {'FaceColor',[0.6 0 0.5]}, ...
        'cursor_inactive_spec', ...
                   {'FaceColor',[0 0 0]}, ...
        'free_trials', 0, ...
        'background', 0.5*[1 1 1], ...
        'color_hit', [0 0.8 0], ...
        'color_miss', [1 0 0], ...
        'color_reject', [0.8 0 0.8], ...
        'color_center', 0.5*[1 1 1], ...
        'center_size', 0.15, ...
        'damping_in_target', 'quadratic', ...
        'target_width', 0.075, ...
        'frame_color', 0.8*[1 1 1], ...
        'punchline', 0, ...
        'punchline_spec', {'Color',[0 0 0], 'LineWidth',3}, ...
        'punchline_beaten_spec', {'Color',[1 1 0], 'LineWidth',5}, ...
        'gap_to_border', 0.02, ...
		    'msg_spec', {'FontSize',0.15}, ...
        'parPort', 1,...
        'changed', 0,...
        'show_score', 0, ...
        'show_rejected', 0, ...
        'show_bit', 0,...
        'log_state_changes',0, ...
        'verbose', 1, ...
        'log',1,...
        'fs', 25, ...
        'status', 'pause', ...
        'pause_msg', 'pause', ...
        'position', VP_SCREEN);
  
  [HH, cfd]= feedback_cursor_arrow_training_init(fig, opt);
  [handles, H]= fb_handleStruct2Vector(HH);
  
  do_set('init', handles, 'cursor_arrow', opt);
  do_set(200);

  memo.stopwatch= 0;
  memo.x= 0;
  memo.degree= 0;
  memo.lastdigit= NaN;
  memo.laststate= NaN;
  memo.laststatus= 'urknall';
  memo.timer= 0;
  memo.lasttick= clock;
  memo.timeeps= 0.25*1000/opt.fs;
  memo.modus= 1;
  state= -1;
end

if opt.changed,
  if ~strcmp(opt.status, memo.laststatus),
    memo.laststatus= opt.status;
    switch(opt.status),
     case 'play',
      do_set(210);
      do_set(H.msg_punch, 'Visible','off');
      state= 0;
     case 'pause',
      do_set(211);
      do_set(H.msg, 'String',opt.pause_msg, 'Visible','on');
      state= -1;
     case 'stop';
      do_set(255);
      do_set(H.msg, 'String','stopped', 'Visible','on');
      state= -2;
    end
  end
end

if opt.changed==1,
  opt.changed= 0;
  do_set(H.msg, opt.msg_spec{:});
  do_set([H.msg_punch], opt.msg_spec{:});
  if ~strcmpi(opt.timeout_policy,'hitiflateral'),
    memo.center_visible= 0;
    do_set(H.center, 'Visible','off');
  else
    memo.center_visible= 1;
    do_set(H.center, 'FaceColor',opt.color_center, 'Visible','on');
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
  if opt.verbose>1,
    fprintf('state: %d\n', state);
  end
end

thisisnow= clock;
time_since_lasttick= 1000*etime(thisisnow, memo.lasttick);
memo.lasttick= thisisnow;

if state==0,
  if memo.timer == 0,
    do_set(H.fixation, 'Visible','off');
  end
  digit= ceil((opt.countdown-memo.timer)/1000);
  if digit~=memo.lastdigit,
    do_set(H.msg, 'String',int2str(digit), 'Visible','on');
    memo.lastdigit= digit;
  end
  if memo.timer+memo.timeeps >= opt.countdown,
    %% countdown terminates: prepare run
    do_set(H.msg, 'Visible','off');
    memo.timer= 0;
    if memo.modus==0,      %% during break inbetween run
      state= 1;
    else
      memo.punch= [-1 1];
      memo.ishit= [];
      memo.rejected= 0;
      nTrials= opt.trials_per_run;
      nBlocks= floor(nTrials/opt.break_every);
      seq= [];
      nClasses= length(opt.classes);
      if mod(opt.break_every,nClasses)
        warning('parameter ''break_every'' should be multiple of nClasses.');
      end
      for bb= 1:nBlocks,
        bseq= floor(linspace(1, nClasses+1, opt.break_every+1));
        bseq(end)= [];
        pp= randperm(opt.break_every);
        seq= cat(2, seq, bseq(pp));
      end
      nLeftOvers= nTrials - length(seq);
      bseq= round(linspace(1, nClasses, nLeftOvers));
      pp= randperm(nLeftOvers);
      seq= cat(2, seq, bseq(pp));
      memo.sequence= [seq, 0];
      memo.trial= -opt.free_trials;
      state= 1;
    end
  else
    memo.timer= memo.timer + time_since_lasttick;
  end
end

if state==1,   %% show a blank screen with fixation cross
  if memo.timer == 0,
    do_set(H.cue, 'Visible','off');
    do_set(H.cursor, 'Visible','off');
    do_set(H.fixation, 'Visible','on');
  end
  if memo.timer+memo.timeeps >= opt.duration_blank,
    memo.timer= 0;
    state= 2;
  end
  memo.timer= memo.timer + time_since_lasttick;
end

if state==2,  %% select goal side and indicate it
  memo.trial_duration= opt.duration + opt.duration_jitter*rand;
  memo.trial= memo.trial + 1;
  if memo.trial==1,
    memo.stopwatch= 0;
  end    
  if memo.trial<1,
    memo.goal= ceil(2*rand);
  else
    memo.goal= memo.sequence(memo.trial);
  end
  do_set(H.cue(memo.goal), 'Visible','on');
  memo.x= 0;
  memo.timer= 0;
  state= 3;
  do_set(memo.goal);
end

if state==3,  %% wait before cursor movement starts
  if memo.timer+memo.timeeps >= opt.duration_before_free,
    state= 4;
  else
    memo.timer= memo.timer + time_since_lasttick;
  end
end

if state==4,  %% in position control, cursor becomes active in center area
  do_set(60);
  if opt.cursor_on,
    do_set(H.fixation, 'Visible','off');
    ud= get(HH.cursor, 'UserData');
    do_set(H.cursor, 'Visible','on', ...
           opt.cursor_active_spec{:}, ...
           'XData',ud.xData, 'YData',ud.yData);
  end
  memo.trialwatch= 0;
  state= 5;
end

if state==5,  %% move cursor until target was hit or time-out
  for ii= 1:3,
    if ~ismember((memo.goal), [opt.trigger_classifier_list{ii}{:}]),
      classifier_factor(ii)= 0;
    else
      if memo.goal==opt.trigger_classifier_list{ii}{1},
        classifier_factor(ii)= -1;
      else
        classifier_factor(ii)= 1;
      end
    end
  end
  evidence_in_favor_of_cue= classifier_factor .* [ctrl1 ctrl2 ctrl3];
  if max(abs(evidence_in_favor_of_cue))==max(evidence_in_favor_of_cue),
    if opt.proportional_control,
      ctrl= min(1, max(evidence_in_favor_of_cue));
    else
      ctrl= 1;
    end
  else
    if opt.proportional_control,
      ctrl= opt.gain_error * max(-1, min(evidence_in_favor_of_cue));
    else
      ctrl= -opt.gain_error;
    end
  end
  if ctrl>0,
    memo.x= memo.x + ctrl/opt.fs/opt.duration_until_hit*1000;
  else
    if memo.x > opt.fallback,
      memo.x= memo.x + ctrl/opt.fs/opt.duration_until_hit*1000;
      memo.x= max(opt.fallback, memo.x);
    end
  end
  memo.x= max(-1-cfd.target_width, min(1+cfd.target_width, memo.x));
  
  trial_terminates= 0;    
  %% timeout
  if memo.trialwatch+memo.timeeps >= memo.trial_duration,
    trial_terminates= 1;
    do_set(29);  %% mark end of the trial
  end
  
  %% trial terminates
  if trial_terminates,
    do_set(H.cursor, opt.cursor_inactive_spec{:});
    if memo.trial>0,
      state= 6;
    end
  end
  memo.trialwatch= memo.trialwatch + time_since_lasttick;
end

if state==6,  %% wait before next trial starts (or game is over)
  if memo.timer+memo.timeeps >= opt.duration_show_selected,
    do_set(H.punchline, opt.punchline_spec{:});
    if memo.trial==length(memo.sequence)-1,
      %% game over
      msg= sprintf('thank you');
      do_set(H.msg, 'String',msg, 'Visible','on');
      if opt.punchline,
        msg= sprintf('punch at  [%d %d]', ...
                     round(100*(memo.punch-sign(memo.punch))/cfd.target_width));
        do_set(H.msg_punch, 'String',msg, 'Visible','on');
      end
      do_set(H.cursor, 'Visible','off');
      do_set(H.fixation, 'Visible','off');
      do_set(H.cue, 'Visible','off');
      do_set(255);
      %       memo.timer= 0;
      %       state= 10;
      state= -1;
    elseif opt.break_every>0 & memo.trial>0 & mod(memo.trial,opt.break_every)==0,
      memo.timer= 0;
      state= 7;
    else
      memo.timer= 0;
      state= 1;
      opt = feedback_cursor_arrow_training(fig, opt, ctrl1, ctrl2, ctrl3);
      memo.timer= memo.timer + time_since_lasttick;
      return;
    end
  else
    memo.timer= memo.timer + time_since_lasttick;
  end
end

if state==7,   %% give a break where the score is (optionally) shown
  if memo.timer == 0,
    do_set(H.center, 'Visible','off');
    do_set(H.cursor, 'Visible','off');
    do_set(H.fixation, 'Visible','off');
    do_set(H.cue, 'Visible','off');
  end
  if memo.timer+memo.timeeps >= opt.duration_break,
    memo.timer= 0;
    state= 8;
  end
  memo.timer= memo.timer + time_since_lasttick;
end

if state==8,   %% score fade-out at the end of the break
  if memo.timer+memo.timeeps > opt.duration_break_fadeout+opt.duration_break_post_fadeout,
    memo.timer= 0;
    if opt.break_endswith_countdown,
      memo.modus= 0;
      state= 0;
    else
      state= 1;
    end
  elseif memo.timer+memo.timeeps <= opt.duration_break_fadeout,
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
    end
  end
  memo.timer= memo.timer + time_since_lasttick;
end

%if state==10,   %% wait before stop recording
%  if memo.timer>=500,
%    bvr_sendcommand('stoprecording');
%    fprintf('EEG recording stopped.\n');
%    state= -1;
%  end
%  memo.timer= memo.timer + time_since_lasttick;
%end
 
memo.stopwatch= memo.stopwatch + time_since_lasttick;

if state>=5,
  ud= get(HH.cursor, 'UserData');
  switch(opt.classes{memo.goal}),
   case 'left',
    do_set(H.cursor, 'XData',ud.xData - abs(memo.x), 'YData',ud.yData);
   case 'right',
    do_set(H.cursor, 'XData',ud.xData + abs(memo.x), 'YData',ud.yData);
   case {'down','foot'},
    do_set(H.cursor, 'YData',ud.yData - abs(memo.x), 'XData',ud.xData);
   case {'up','tongue'},
    do_set(H.cursor, 'YData',ud.yData + abs(memo.x), 'XData',ud.xData);
   otherwise,
    error(sprintf('unknown value for opt.classes: <%s>', ...
                  opt.classes{iDir}));
  end
end

do_set('+');
