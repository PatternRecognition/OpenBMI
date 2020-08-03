function opt = feedback_tappedCross(fig, opt, x, y);
%
% opt:
%  .command_mode - {'none', 'letter', 'target'}
%  .show_fixation - {0,1}, show fixation cross?
%  .fixation_size - [sz, lw], sz: size relative to xLim, lw: lineWidth

persistent state cursor tail target paused
persistent counter running_time waiting touch lasttouch highlight_time
persistent iCursor iFix iCounter iMsg iStim iPatch iTail

global lost_packages

if ~isfield(opt,'reset') 
  opt.reset = 1;
end

if opt.reset,
  opt= set_defaults(opt, ...
                    'direction', 'horizontal', ...
                    'command_mode', 'none', ...
                    'cursor_active_type', '+', ...
                    'cursor_inactive_type', '+', ...
                    'cursor_active_size', [40 5], ...
                    'cursor_inactive_size', [40 5], ...
                    'cursor_active_color', [0 0 0], ...
                    'cursor_inactive_color', 0.4*[1 1 1], ...
                    'tail_length', 3, ...
                    'tail_linewidth', 2, ...
                    'tail_marker', '.', ...
                    'tail_markersize', 20, ...
                    'tail_color', 0.4*[1 1 1], ...
                    'patch_x', [0.25 0.5], ...
                    'patch_y', [0 0.25], ...
                    'stimulus', 'LR', ...
                    'stimulus_fontsize', 0.15, ...
                    'stimulus_color', 'k', ...
                    'stimulus_pos', [0 0], ...
                    'freeregion', [-1 -1 2 2], ...
                    'freeregion_visible', 'off', ...
                    'freeregion_color', 0.5*[1 1 1], ...
                    'freeing_time', 500, ...
                    'background_color', 0.8*[1 1 1], ...
                    'hit_color', [1 0 0; 0 0.9 0], ...
                    'miss_color', [1 0 0; 0 0.9 0], ...
                    'target_color', [0.8 0 0; 0 0.7 0], ...
                    'nontarget_color', [0.8 0.5 0.5; 0.5 0.7 0.5], ...
                    'fixation_visible', 'on', ...
                    'fixation_color', 0*[1 1 1], ...
                    'fixation_size', [1 1], ...
                    'fixation_position', [0 0], ...
                    'hit_highlight_time', 500, ...
                    'time_before_active', 100, ...
                    'counter_visible', 'off', ...
                    'counter_pos', [-0.86 -0.9], ...
                    'counter_fontsize', 0.06, ...
                    'matchpoints', 100, ...
                    'msg_fontsize', 0.15, ...
                    'countdown', 3000, ...
                    'fs', 25, ...
                    'status', 'pause', ...
                    'log', 0, ...
                    'changed', 0, ...
                    'position', get(fig,'position'),...
                    'parPort',1);

  [hh]= feedback_tappedCross_init(fig, opt);
  do_set('init',hh, 'tappedCross',opt);
                
  % These are the handle indices according to feedback_tappedCross_init:
  iCursor = 3;
  iFix = 4:5;
  iCounter = 6;
  iMsg = 7;
  iStim= 8;
  iPatch = 9:10;
  iTail = 11:length(hh);
  
  tail= zeros(opt.tail_length+1, 2);
  counter= [0 0];
  opt.reset = 0;
  waiting= 0;
  paused= 0;
  state= 0;
end


if opt.changed==1 & opt.log==1,
  opt.changed= 0;
  % write log?
end

if strcmp(opt.status, 'pause') & ~paused,
  do_set(iMsg, 'string','pause', 'visible','on');
  do_set(211);
  paused= 1;
end

if paused,
  if strcmp(opt.status, 'play'),
    do_set(209);
    waiting= 0;
    paused= 0;
    state= 0;
  else
    return;
  end
end

switch(state),
 case 0,  %% countdown to start
  waiting= waiting + 1000/opt.fs;
  if waiting>=opt.countdown,
    do_set(iMsg, 'visible','off');
    do_set(210);
    running_time= 0;
    state= 1;
  else
    str= sprintf('start in %.0f s', ceil((opt.countdown-waiting)/1000));
    do_set(iMsg, 'string',str, 'visible','on');
  end
 case 1,  %% choose target (if desired)
  do_set(iCursor, 'marker', opt.cursor_active_type, ...
         'markerSize',opt.cursor_active_size(1), ...
         'lineWidth',opt.cursor_active_size(2), ...
         'markerEdgeColor', opt.cursor_active_color);
  if ~strcmp(opt.command_mode, 'none'),
    target= ceil(2*rand);
    do_set(target);
    switch(opt.command_mode),
     case 'letter',
      do_set(iStim, 'string',opt.stimulus(target), 'visible','on');
     case 'target',
      do_set(iPatch(target), 'faceColor',opt.target_color(target,:));
    end
  end
  state= 2;
 case 2,  %% wait for target hit
  touch= 0;
  if (abs(x)>=opt.patch_x(1) & y>=opt.patch_y(1) & ...
      ~(y<=opt.patch_y(2)-abs(x)*diff(opt.patch_x)/diff(opt.patch_y))),
    touch= 1.5+sign(x)/2;
  end
  if touch>0,
    if ~strcmp(opt.command_mode, 'none'),
      ishit= (touch==target);
      counter(2-ishit)= counter(2-ishit)+1;
      do_set(iCounter, 'string', sprintf('%d:%d', counter));
      do_set(10+ishit);  %% 10:hit, 11:miss
    else
      ishit= 1;
      do_set(50+touch);  %% 51: left, 52: right
    end
    if ishit,
      do_set(iPatch(touch), 'faceColor',opt.hit_color(touch,:));
    else
      do_set(iPatch(touch), 'faceColor',opt.miss_color(touch,:));
    end
    if sum(counter)==opt.matchpoints,  %% game over
      if ~strcmp(opt.command_mode, 'none'),
        bpd= bitrate(counter(1)/sum(counter), opt.targets);
        bpm= bpd*sum(counter)/running_time*60000;
        str= sprintf('%.1f bits/min', bpm);
        do_set(iMsg, 'string',str, 'visible','on');
      end
      opt.status= 'pause';
      paused= 1;
    else
      if strcmp(opt.command_mode, 'letter'),
        do_set(iStim, 'visible','off');
      end
      untouched= 3-touch;
      do_set(iPatch(untouched), ...
             'faceColor',opt.nontarget_color(untouched,:));
      do_set(iCursor, 'marker', opt.cursor_inactive_type, ...
             'markerSize',opt.cursor_inactive_size(1), ...
             'lineWidth',opt.cursor_inactive_size(2), ...
             'markerEdgeColor', opt.cursor_inactive_color);
      highlight_time= opt.hit_highlight_time;
      lasttouch= touch;
      if opt.freeing_time>0,
        waiting= opt.freeing_time;
        state= 3;
      else
        waiting= opt.time_before_active;
        state= 4;
      end
    end
  else
    running_time= running_time + 1000/opt.fs*(1+lost_packages);
  end
 case 3,  %% wait for cursor to stay in free region
  infreeregion= pointinrect([x y], opt.freeregion);
  if (abs(x)>=opt.patch_x(1) & y>=opt.patch_y(1) & ...
      ~(y<=opt.patch_y(2)-abs(x)*diff(opt.patch_x)/diff(opt.patch_y))),
    infreeregion= 0;
  end
  if infreeregion,
    waiting= waiting - 1000/opt.fs;
    if waiting<=0,
      do_set(59);  %% cursor is set free
      waiting= opt.time_before_active;
      state= 4;
    end
  else
    waiting= opt.freeing_time;
  end
 case 4,  %% wait before choosing next target
  waiting= waiting - 1000/opt.fs;
  if waiting<0,
    state= 1;
  end
end

if highlight_time>0,
  highlight_time= highlight_time - 1000/opt.fs;
  if highlight_time<=0,
    do_set(iPatch(lasttouch), 'faceColor',opt.nontarget_color(lasttouch,:));
  end
end

if strcmp(opt.direction, 'horizontal'),
  cursor= [x, y];
else
  cursor= [y, x];
end
if opt.tail_length>0,
  tail= [cursor; tail(1:end-1,:)];
  for kk= 1:opt.tail_length,
    do_set(iTail(kk), 'xData',tail([kk kk+1],1), 'yData',tail([kk kk+1],2));
  end
end

do_set(iCursor, 'xData',cursor(1), 'yData',cursor(2));
do_set('+');
