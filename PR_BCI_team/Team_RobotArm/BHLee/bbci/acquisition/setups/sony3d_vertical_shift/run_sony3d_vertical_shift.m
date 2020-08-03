%% Run the Sony 3D Vertical-Shift Study

if isempty(VP_CODE)
VP_CODE=input('Enter VP_CODE used in a previous experiment \nin quotation marks and press <ENTER>. \nIf not known: Only press <ENTER>');
acq_makeDataFolder;
end

% Check
if isempty(VP_CODE)
 error('Define VP_CODE!');
end
if isempty(TODAY_DIR)
 error('Define TODAY_DIR!');
end


pause_during_pic = 4.5;
pause_before_pic = 1;
pause_after_pic  = 1;
video_overlap    = 10;
num_repetitions  = 12;

standard_markers;

resources_list_vertical_shift; 
setup_sony3d

for i=1:size(videos,2)
  fprintf('%d: %s\n', i, videos{3,i});
end
which_video = input('Which video? ');
if isempty(which_video)
  return
end
video_fname = videos{1,which_video};
video_duration = videos{2,which_video};

rep_start = input('Start with which run? (In case the subject has completed some already) [1]? ');
if isempty(rep_start)
  rep_start=1;
end

%calculate duration
num_pics = numel(file_names);

approx_time_to_answer = 5;
approx_pause_before_rep = 5;
rep_dur = (pause_before_pic + pause_during_pic + pause_after_pic + approx_time_to_answer) ...
          * num_pics;
dur = (approx_pause_before_rep + rep_dur) * num_repetitions ...
      + video_duration + video_overlap * num_repetitions;

fprintf('\n%d repetitions of %d pics, each repetition %.1f min, %.1f min video in between,\n', num_repetitions, num_pics, rep_dur/60, (video_duration/(num_repetitions+1)+video_overlap)/60); 
fprintf('approximate duration of the experiment: %.1f min\n\n', dur/60);

% At first, select proper adjustment for video presentation:
show_video_snippet(video_fname, 1, 1, 1, 1);
fprintf('Select: "2D Monoskopisch" --> Enter\n'); pause;


if rep_start==1
    

    %show intro
    pnet(tcp_conn, 'printf', 'msg intro0\n');             pause;
    
    pnet(tcp_conn, 'printf', 'msg intro1\n');             pause;
    pnet(tcp_conn, 'printf', 'pic %s\n', file_names{22}); pause(pause_during_pic);
    pnet(tcp_conn, 'printf', 'msg intro1q\n');            quality_input();
    
    pnet(tcp_conn, 'printf', 'msg intro3\n');             pause;
    pnet(tcp_conn, 'printf', 'pic %s\n', file_names{28}); pause(pause_during_pic);
    pnet(tcp_conn, 'printf', 'msg intro3q\n');            quality_input();

    pnet(tcp_conn, 'printf', 'msg intro2\n');             pause;
    pnet(tcp_conn, 'printf', 'pic %s\n', file_names{24}); pause(pause_during_pic);
    pnet(tcp_conn, 'printf', 'msg intro2q\n');            quality_input();

    % Again for the cubes:
    pnet(tcp_conn, 'printf', 'msg intro0cubes\n');         pause;
    pnet(tcp_conn, 'printf', 'msg intro1cubes\n');             pause;
    pnet(tcp_conn, 'printf', 'pic %s\n', file_names{8}); pause(pause_during_pic);
    pnet(tcp_conn, 'printf', 'msg intro1q\n');            quality_input();
    
    pnet(tcp_conn, 'printf', 'msg intro3\n');             pause;
    pnet(tcp_conn, 'printf', 'pic %s\n', file_names{14}); pause(pause_during_pic);
    pnet(tcp_conn, 'printf', 'msg intro3q\n');            quality_input();

    pnet(tcp_conn, 'printf', 'msg intro2\n');             pause;
    pnet(tcp_conn, 'printf', 'pic %s\n', file_names{10}); pause(pause_during_pic);
    pnet(tcp_conn, 'printf', 'msg intro2q\n');            quality_input();

    % Introduction 2D:
    pnet(tcp_conn, 'printf', 'msg intro2Dimages\n');      pause;
    pnet(tcp_conn, 'printf', 'pic %s\n', file_names{18}); pause(pause_during_pic);
    pnet(tcp_conn, 'printf', 'pic %s\n', file_names{4}); pause(pause_during_pic);

end




fprintf('Waiting for subject to press enter...\n');
pnet(tcp_conn, 'printf', 'msg press_enter_for_experiment\n');
pause

for i = rep_start:num_repetitions
  
  %start recording
  bvr_startrecording(['sony3d_' VP_CODE], 'impedances', 0);
  pause(3);
  
  %get prepared, press enter when ready
  pnet(tcp_conn, 'printf', 'msg press_enter_for_run\n');
  fprintf('Press enter to start run nr. %d/%d!\n', i, num_repetitions);
  pause
  
  % For test without entering numbers:pause(2)

  
  
  
  
  p = randperm(num_pics);

  for j=1:num_pics
    
    pnet(tcp_conn, 'printf', 'msg black\n');
    pause(pause_before_pic);
    
    ppTrigger(file_markers(p(j)) + marker_start);
    pnet(tcp_conn, 'printf', 'pic %s\n', file_names{p(j)});
    pause(pause_during_pic);
    ppTrigger(file_markers(p(j)) + marker_end);
    
    %please enter score
    
    pnet(tcp_conn, 'printf', 'msg black\n');
    pause(pause_after_pic);

    ppTrigger(marker_user_entry);
    pnet(tcp_conn, 'printf', 'msg question\n');
    
    user_entry = quality_input(); 
    
    % For test without entering numbers:
    %%user_entry = ceil(3.*rand(1,1)); pause(ceil(5.*rand(1,1)));
    
    ppTrigger(marker_user_entry + user_entry);
  end

  pnet(tcp_conn, 'printf', 'msg loading_video\n');
  %show video for relaxation
  ppTrigger(marker_video + marker_start);
  
  %show_video_snippet(video_fname, video_duration, video_overlap, i, num_repetitions+1);
  show_video_snippet(video_fname, video_duration, video_overlap, i, num_repetitions);
  
  ppTrigger(marker_video + marker_end);
  
  pause(1)
  bvr_sendcommand('stoprecording');
  pause(1)
  
end

%show_video_snippet(video_fname, video_duration, video_overlap, num_repetitions+1, num_repetitions+1);
pnet(tcp_conn, 'close');
fprintf('done!\n');
