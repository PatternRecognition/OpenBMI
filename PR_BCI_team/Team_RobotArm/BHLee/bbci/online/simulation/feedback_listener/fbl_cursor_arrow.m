function opt = fbl_cursor_arrow(fig, opt, ctrl);
%FBL_CURSOR_ARROW - BBCI Feedback Listener: 1D Cursor Movement with Arrow Cues
%
%Synopsis:
% OPT= fbl_cursor_arrow(FIG, OPT, CTRL)
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
%   60: cursor starts movement
%   70: start of adaptation phase 1 (rotating cursor)
%   71: end of adaptation phase 1
%  200: init of the feedback
%  210: game status changed to 'play'
%  211: game status changed to 'pause'
%  254: game ends
%
%See:
%  feedback_cursor_arrow, feedback_cursor_arrow_init

% Author(s): Benjamin Blankertz, Sep-2008

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
        'classes', {'left','right'}, ...
        'duration', 4000, ...
        'duration_jitter', 0, ...
        'duration_before_free', 1000, ...
        'duration_show_selected', 2000, ...
        'duration_blank', 2000, ...
        'duration_until_hit', 2000, ...
        'duration_break', 15000, ...
        'duration_break_fadeout', 1000, ...
        'duration_break_post_fadeout', 1000, ...
        'touch_terminates_trial', 0, ...
        'break_every', 20, ...
        'break_show_score', 1, ...
        'break_endswith_countdown', 1, ...
        'timeout_policy', 'hitiflateral', ...
        'rate_control', 1, ...
        'cursor_on', 1, ...
        'response_at', 'none', ...
        'trials_per_run',100,...
        'adapt_trials', 0, ...
        'cursor_active_spec', ...
                   {'FaceColor',[0.32 0 0.4]}, ...
        'cursor_inactive_spec', ...
                   {'FaceColor',[0 0 0]}, ...
        'free_trials', 0, ...
        'adapt_trials', 0, ...
        'background', 0.5*[1 1 1], ...
        'color_hit', [0 0.8 0], ...
        'color_miss', [1 0 0], ...
        'color_reject', [0.8 0 0.8], ...
        'color_center', 0.5*[1 1 1], ...
        'center_size', 0.3, ...
        'damping_in_target', 'quadratic', ...
        'target_width', 0.075, ...
        'frame_color', 0.8*[1 1 1], ...
        'punchline', 0, ...
        'punchline_spec', {'Color',[0 0 0], 'LineWidth',3}, ...
        'punchline_beaten_spec', {'Color',[1 1 0], 'LineWidth',5}, ...
        'gap_to_border', 0.02, ...
		    'msg_spec', {'FontSize',0.15}, ...
		    'points_spec', {'FontSize',0.065}, ...
        'parPort', 1,...
        'changed', 0,...
        'show_score', 0, ...
        'show_rejected', 0, ...
        'show_bit', 0,...
        'log_state_changes',0, ...
        'log',0,...
        'fs', 25, ...
        'status', 'pause', ...
        'position', VP_SCREEN);
  
  if ~opt.touch_terminates_trial & isdefault.punchline,
    [opt, isdefault]= opt_overrideIfDefault(opt, isdefault, ...
                                            'punchline', 1);
  end
  if ~ismember(opt.timeout_policy,{'hitiflateral','reject'}) ...
        & isdefault.show_rejected,
    [opt, isdefault]= opt_overrideIfDefault(opt, isdefault, ...
                                            'show_rejected', 0);
  end
  
  [HH, cfd]= feedback_cursor_arrow_init(fig, opt);
  [handles, H]= fb_handleStruct2Vector(HH);
  
  do_set('init', handles, 'cursor_arrow', opt);

  memo.stopwatch= 0;
  memo.x= 0;
  memo.degree= 0;
  memo.laststate= NaN;
  memo.laststatus= 'urknall';
  memo.timer= 0;
  if opt.adapt_trials>0,
    memo.modus= 2;
  else
    memo.modus= 1;
  end
  
  state= 0;
  do_set(H.cue, 'Visible','off');
  do_set(H.cursor, 'Visible','off');
  do_set(H.fixation, 'Visible','on');
end

if ~opt.rate_control & state>-2,
  memo.x= ctrl;
  memo.x= max(-1, min(1, memo.x));
end

if state==0,  %% init
  memo.trial= 0;
  memo.ishit= [];
  memo.rejected= 0;
  memo.orig_result= [];
  memo.withintrial= 0;
  memo.punch= [-1 1];
  memo.touched_once= 0;
  if memo.modus==1,
    memo.sequence= zeros(1, opt.trials_per_run);
  else
    memo.sequence= zeros(1, 2*opt.adapt_trials);
  end
  state= 1;
end

[toe,timeshift] = adminMarker('query', [-39 0]);


mrk_cue= {1, 2};
mrk_hit= {11, 12};
mrk_miss= {21, 22, 23};
mrk_reject= {24, 25};
relevant= intersect([mrk_cue{:}, mrk_hit{:}, mrk_miss{:}, mrk_reject{:}], ...
                    toe);
iRelevant= find(ismember(toe, relevant), 1, 'last');
if ~isempty(iRelevant),
%  keyboard
  orig_result= 0;
  switch(toe(iRelevant)),
   case(mrk_cue),
    memo.sequence(memo.trial+1)= find(toe(iRelevant)==[mrk_cue{:}]);
    state= 2;
   case(mrk_hit),
    orig_result= 1;
   case(mrk_miss),
    orig_result= 2;
   case(mrk_reject),
    orig_result= 3;
  end
  if orig_result>0,
    memo.orig_result= [memo.orig_result orig_result];
    fprintf('orignal score: %03d : %03d  (%d rejected)\n', ...
            sum(memo.orig_result==1), ...
            sum(memo.orig_result==2), ...
            sum(memo.orig_result==3));
    if memo.withintrial,
      state= 99;
    end
  end
end

if state==2,  %% select goal side and indicate it
  memo.withintrial= 1;
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
end

if state==3,  %% wait before cursor movement starts
  if memo.timer>=opt.duration_before_free,
    state= 4;
  else
    memo.timer= memo.timer + 1000/opt.fs;
  end
end

if state==4,  %% in position control, cursor becomes active in center area
  if opt.rate_control | abs(memo.x)<opt.center_size,
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
end

if state==5,  %% move cursor until target was hit or time-out
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
%      do_set(10*(2-ishit)+memo.selected);
    elseif ~memo.touched_once,
      memo.touched_once= 1;
      memo.preselected= sign(memo.x)/2 + 1.5;
      ishit= (memo.preselected==memo.goal);
%      do_set(30+10*(2-ishit)+memo.preselected);
    end
  end
  
  %% timeout
  if memo.trialwatch >= memo.trial_duration,
    state= 99;
  elseif trial_terminates,
    state= 100;
  end
  memo.trialwatch= memo.trialwatch + 1000/opt.fs;
end

if state== 99;
  %% time out
    switch(lower(opt.timeout_policy)),
     case 'miss',
      memo.selected= [];
      ishit= 0;
%     do_set(23);  %% reject by timeout
     case 'reject',
      memo.selected= [];
      ishit= -1;
%     do_set(24);  %% reject by timeout
     case 'lastposition',
      memo.selected= sign(memo.x)/2 + 1.5;
      ishit= (memo.selected==memo.goal);
%     do_set(10*(2-ishit)+memo.selected);
     case 'hitiflateral',
      if abs(memo.x)>opt.center_size,
        %% cursor is lateral enough (outside center): count as hit
        memo.selected= sign(memo.x)/2 + 1.5;
        ishit= (memo.selected==memo.goal);
%       do_set(10*(2-ishit)+memo.selected);
      else
        %% cursor is within center area: count as reject
        memo.selected= [];
        ishit= -1;
%       do_set(25);  %% reject by timeout and position
      end
     otherwise
      error('unknown value for OPT.timeout_policy');
    end
  state= 100;  
end

if state==100,
  %% trial terminates: show response and update score
  memo.winthintrial= 0;
  do_set(H.cursor, opt.cursor_inactive_spec{:});
  if memo.trial>0,
    if ishit==-1,
      memo.rejected= memo.rejected + 1;
    end      
    memo.ishit(memo.trial)= (ishit==1);
    nHits= sum(memo.ishit(1:memo.trial));
    nMisses= memo.trial - memo.rejected - nHits;
    fprintf(' replay score: %03d : %03d  (%d rejected)\n', nHits, nMisses, ...
            memo.rejected);
    do_set(H.points(1), 'String',['hit: ' int2str(nHits)]);
    do_set(H.points(2), 'String',['miss: ' int2str(nMisses)]);
    do_set(H.rejected_counter, 'String',['rej: ' int2str(memo.rejected)]);
    if ishit==1 & ~opt.touch_terminates_trial,
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
    memo.H_indicator= H.cue(memo.goal);
    ind_prop= 'FaceColor';
   case 'cursor',
    memo.H_indicator= H.cursor;
    ind_prop= 'Color';
   case 'none',
    memo.H_indicator= [];
   otherwise,
    warning('value for property ''response at'' unrecognized.');
    memo.H_indicator= H.target(memo.selected);
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
  state= 6;
end

if state==6,  %% wait before next trial starts (or game is over)
  if memo.timer >= opt.duration_show_selected,
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
    do_set(H.punchline, opt.punchline_spec{:});
    if memo.trial==length(memo.sequence)-1 & memo.modus==1,
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
        msg= sprintf('game over');
        do_set(H.msg, 'String',msg, 'Visible','on');
      end
      if opt.punchline,
        msg= sprintf('punch at  [%d %d]', ...
                     round(100*(memo.punch-sign(memo.punch))/cfd.target_width));
        do_set(H.msg_punch, 'String',msg, 'Visible','on');
      end
      if opt.rate_control,
        do_set(H.cursor, 'Visible','off');
      else
        do_set(H.cursor, opt.cursor_inactive_spec{:});
      end
      do_set(H.fixation, 'Visible','off');
      do_set(H.cue, 'Visible','off');
%      do_set(254);
      %       memo.timer= 0;
      %       state= 10;
      state= -1;
    else
      memo.timer= 0;
      state= 1;
      do_set(H.cue, 'Visible','off');
      do_set(H.cursor, 'Visible','off');
      do_set(H.fixation, 'Visible','on');
      if memo.modus==2 & memo.trial==2*opt.adapt_trials,
        fprintf('Supposing adaptation period finished.\n');
        memo.modus= 1;
        state= 0;
      end
%      opt = fbl_cursor_arrow(fig, opt, ctrl);
      return;
    end
  else
    memo.timer= memo.timer + 1000/opt.fs;
  end
end

memo.stopwatch= memo.stopwatch + 1000/opt.fs;

ud= get(HH.cursor, 'UserData');
if memo.x<0,
  iDir= 1;
else
  iDir= 2;
end
switch(opt.classes{iDir}),
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

do_set('+');
