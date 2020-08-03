if isempty(VP_CODE)
  error('Define VP_CODE!');
end

if isempty(TODAY_DIR)
  today_vec = clock;
  TODAY_DIR = [EEG_RAW_DIR VP_CODE sprintf('_%02d_%02d_%02d/', today_vec(1)-2000, today_vec(2:3))];
  mkdir_rec(TODAY_DIR)
end

%% run scene change experiment
resources_scene_change;

% unique sequence for each VP
rand('seed', sum(VP_CODE));

pause_during_movie = 20;
pause_after_input = 3;

standard_markers;
setup_sony3d

bvr_startrecording(['scenechange_' VP_CODE], 'impedances', 0);
pnet(tcp_conn, 'printf', 'msg press_enter_for_run\n');
pause;
for stim = randperm(numel(scenes))
  pnet(tcp_conn, 'printf', 'msg black\n');
  ppTrigger(black_marker);
  pause(pause_after_input);
  pnet(tcp_conn, 'printf', 'vid %s', scenes{stim});
  ppTrigger(stim_marker(stim));
  pause(pause_during_movie)
  pnet(tcp_conn, 'printf', 'msg question_scene_change\n');
  ppTrigger(question_marker);
  user_answer = scene_change_input();
  ppTrigger(question_marker + user_answer)
end

% pnet(tcp_conn, 'printf', 'msg black\n');
% ppTrigger(end_marker);
% pause(pause_before_movie);
    
bvr_sendcommand('stoprecording');
pause(1)
pnet(tcp_conn, 'close');
fprintf('Done!\n');
