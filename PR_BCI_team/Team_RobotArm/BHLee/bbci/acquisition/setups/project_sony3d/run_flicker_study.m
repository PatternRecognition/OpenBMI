if isempty(VP_CODE)
  error('Define VP_CODE!');
end

if isempty(TODAY_DIR)
  
  today_vec = clock;
  TODAY_DIR = [EEG_RAW_DIR VP_CODE sprintf('_%02d_%02d_%02d/', today_vec(1)-2000, today_vec(2:3))];
  mkdir_rec(TODAY_DIR)
end

%% run flicker experiment
resources_list_flicker;

% unique sequence for each VP
rand('seed',sum(VP_CODE));

pause_during_pic = 10;
pause_before_pic = 2;

standard_markers;
setup_sony3d

bvr_startrecording(['sonyflicker_' VP_CODE], 'impedances', 0);

for f = randperm(numel(freqs))
    pnet(tcp_conn, 'printf', 'msg press_enter_for_run\n');
    fprintf('Set the screen''s frequency to %d Hz and press ENTER\n', freqs(f));
    pause;
    for b = randperm(numel(bright))
    pnet(tcp_conn, 'printf', 'msg press_enter_for_run\n');
    fprintf('Set the goggles'' brightness to %s and press ENTER\n', bright{b});
    pause;
        for s = randperm(numel(stimuli))
            pnet(tcp_conn, 'printf', 'msg black\n');
            ppTrigger(marker_black);
            pause(pause_before_pic);
            if s>9
                pnet(tcp_conn, 'printf', 'jpg %s\n', stimuli{s});
            else
                pnet(tcp_conn, 'printf', 'msg %s\n', stimuli{s});
            end
            ppTrigger(markers(f,b,s));
            pause(pause_during_pic);
        end
    end
end
pnet(tcp_conn, 'printf', 'msg black\n');
ppTrigger(marker_black);
pause(pause_before_pic);
    
bvr_sendcommand('stoprecording');
pause(1)
pnet(tcp_conn, 'close');
fprintf('Done!\n');
