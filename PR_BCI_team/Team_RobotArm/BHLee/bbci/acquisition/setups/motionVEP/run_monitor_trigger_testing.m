

%% load workspace:
bvr_sendcommand('loadworkspace', 'ActiCap_VisualSetup_linked_mastoids_5000Hz');

%% Do it like in the Calibration
startup_pyff
fprintf('Press <RETURN> to start monitor trigger test.\n'), pause;

RUN_START = 252;
END_LEVEL2 = 245;
currentSpeller = 'CenterSpellerVE';
all_letters = 'BBCI.';

monitor = {'TFT', 'CRT'};
vp_screen = {[0 0 1280 1024], [0 0 1280 1024]};

color = {'color', 'bw'};
stimuli_color = {{[1.0 0.0 0.0], ...
                  [0.0 0.8 0.0], ...
                  [0.0 0.0 1.0], ...
                  [1.0 1.0 0.0], ...
                  [1.0 0.0 0.7], ...
                  [0.9 0.9 0.9]}, ...
                 {[1 1 1], ...
                  [1 1 1], ...
                  [1 1 1], ...
                  [1 1 1], ...
                  [1 1 1], ...
                  [1 1 1]}};
letter_color = {ones(1,3)*0.5, ones(1,3)};
osc_color = {[0.7 0.7 0.7], [1 1 1]};

programs = {'fb','fb_et','fb_et_cfy'};

for mm=1:numel(monitor),
  if mm>1
    fprintf('Change and configure monitor, then press <RETURN>!'), pause
  end
  VP_SCREEN = vp_screen{mm};
  
  for pp=1:numel(programs),

    for cc=1:numel(color),
      
      %% Turn on eyetracker:
      if ~isempty(strfind(programs{pp},'et'))
        fprintf('\n!!! Turn on Eye-Tracker and press <RETURN>!!!\n'); pause;
      end

      %% Feedback settings for the speller
      fb= struct();
      fb.log_filename= {'s',[TODAY_DIR 'monitor_test_' monitor{mm} '_' color{cc} '_' programs{pp} '.log']};
      
      fb.debug= {'i', 0};
      fb.offline= {'i', isempty(strfind(programs{pp},'cfy'))};

      % Visual settings
      fb.screenPos = {'i', int16(VP_SCREEN)};
      fb.fullscreen = {'i',1};
      fb.stimuli_colors = {'f', stimuli_color{cc}};
      fb.letter_color = {'f', letter_color{cc}};
      fb.osc_color = {'f', osc_color{cc}};

      % Eyetracker
      fb.use_eyetracker = {'i', ~isempty(strfind(programs{pp},'et'))};
      fb.et_duration = {'i',100};
      fb.et_range = {'i',200000};       % Maximum acceptable distance between target and actual fixation
      fb.et_range_time = {'i',200}; 
      
      % Timing and trials
      tframe = 0.016;    % time for one frame
      fb.stimulus_duration = {'f',.1 -tframe *.1};  % subtract 0.1*frame to prevent frameskips
      fb.interstimulus_duration = {'f',.1-tframe *.1};  % ISI
      fb.animation_time = {'f',1.5 -tframe *.1};
      fb.nCountdown = {'i',3};
      fb.nr_sequences = {'i',100};
      fb.min_dist = {'i',2};

      % Init Speller
      pause(.01)
      send_xmlcmd_udp('interaction-signal', 's:_feedback', currentSpeller,'command','sendinit');
      pause(1)

      %% Send settings
      fb_names = fieldnames(fb);
      for fb_idx = 1:length(fb_names),
        send_xmlcmd_udp('interaction-signal', strcat(fb.(fb_names{fb_idx}){1}, ':', fb_names{fb_idx}), fb.(fb_names{fb_idx}){2});
        pause(0.01);
      end
        

      %% Start test!
      send_xmlcmd_udp('interaction-signal', 's:TODAY_DIR',TODAY_DIR, 's:VP_CODE',VP_CODE, ...
        's:BASENAME', ['monitor_test_' monitor{mm} '_' color{cc} '_' programs{pp} '_']);
      pause(5)
      
      if isempty(strfind(programs{pp},'cfy')),
        %% offline:
        
        send_xmlcmd_udp('interaction-signal', 'command', 'play')
        % Initialisiere ersten Buchstaben (fuer offline copy spelling)
        fprintf('Waiting for RUN_START marker.\n');
        stimutil_waitForMarker(['S' num2str(RUN_START)]);
        send_xmlcmd_udp('interaction-signal', 's:_offline_new_letter',all_letters(1));
        % Go through all letters
        for ll=2:numel(all_letters)
          stimutil_waitForMarker(['S' num2str(END_LEVEL2)]);
          send_xmlcmd_udp('interaction-signal', 's:_offline_new_letter',all_letters(ll));
        end
        pause(1)
        stimutil_waitForMarker(['S' num2str(END_LEVEL2)]);
        
      else
        %% online:
        
        bbci= bbci_default;
        bbci.save_name= strcat(EEG_RAW_DIR, 'VPiac_10_05_14/bbci_classifier_', currentSpeller, '_', VP_CODE);
%         bbci.quit_marker= [END_LEVEL2 4 5];
        send_xmlcmd_udp('interaction-signal', 'command', 'play');
        bbci_bet_apply(bbci.save_name, 'bbci.feedback','ERP_Speller', 'bbci.fb_port', 12345);
      end
      

      fprintf('Test finished.\n')
      bvr_sendcommand('stoprecording');
      send_xmlcmd_udp('interaction-signal', 'command', 'stop'), pause(1)

    end
  end
end

send_xmlcmd_udp('interaction-signal', 'command', 'quit');


