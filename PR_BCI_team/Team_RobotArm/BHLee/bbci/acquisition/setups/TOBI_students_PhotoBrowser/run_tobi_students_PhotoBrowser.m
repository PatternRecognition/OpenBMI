fprintf(' Willkommen zum Photobrowser Experiment! \r  \r  Wir w√ºnschen Ihnen Viel Spaﬂ! \r ');
pause;
fprintf(' Kappe wird gesetzt! \r ');

% setup udp ports
send_xmlcmd_udp('init', '127.0.0.1', 12345);

condition_list_fname = {'brightness'  'scale' 'rotation' 'invert' 'mask' 'combination'}
%% Demolauf, VP die verschiedenen Effekte vorstellen fuer den Fragebogen! 
  disp('Proberchlauf starten?')
  %pyff('startup');
  pause;

  kondis=[1 2 3 4 5 6];
            
  send_xmlcmd_udp('interaction-signal', 's:_feedback', 'p300_photobrowser','command','sendinit');
        demo.screen_width = 1920;                                    % window width in pixels
		demo.screen_height = 1200;                                    % window height in pixels
		demo.screen_x = -1920;                                          % x position of left top corner of the window in pixels
		demo.screen_y = 0;                                          % y position of left top corner of the window in pixels
  
    for c = 1:6
       
        switch kondis(c);
                case 1
                    disp('Flash')
                    demo.flash_enable=true;
                    demo.scaling_enable=false;
                    demo.rotation_enable=false;
                    demo.invert_enable=false;
                    demo.mask_enable=false;
                    
                case 2
                   disp('Scale')
                    demo.flash_enable=false;
                    demo.scaling_enable=true;
                    demo.rotation_enable=false;
                    demo.invert_enable=false;
                    demo.mask_enable=false;
                    
                case 3
                   disp('Rotation')
                    demo.flash_enable=false;
                    demo.scaling_enable=false;
                    demo.rotation_enable=true;
                    demo.invert_enable=false;
                    demo.mask_enable=false;
                case 4
                   disp('Invert') 
                    demo.flash_enable=false;
                    demo.scaling_enable=false;
                    demo.rotation_enable=false;
                    demo.invert_enable=true;
                    demo.mask_enable=false;
                    
               case 5
                    disp('Mask')
                    demo.flash_enable=false;
                    demo.scaling_enable=false;
                    demo.rotation_enable=false;
                    demo.invert_enable=false;
                    demo.mask_enable=true;
                    
               case 6
                    disp('Combination')
                    demo.flash_enable=true;
                    demo.scaling_enable=true;
                    demo.rotation_enable=true;
                    demo.invert_enable=false;
                    demo.mask_enable=true;
                    
               otherwise
                   disp('Unknown condition')
        end
        
        % wichtige Std parameter setzen (numSeq, numTrials...)
        demo.num_iterations_menge = [2, 4, 10, 4, 2];
        demo.inter_trial_duration 					= 4000;
		demo.stimulus_duration 						= 100;
        demo.inter_stimulus_duration 				= 100;
        demo.post_cue_presentation_pause_duration 	= 2000;
        demo.display_selection_enable               = false;
        demo.num_blocks                             = 1;
        demo.num_trials                             = 5;
        
        
        % Erste Initialisierung von Pyff mit demo
       
       pause(3);
       
        % SEND demo struct
            demoOpts = fieldnames(demo); 
            
            for optId = 1:length(demoOpts),
                if isnumeric(getfield(demo, demoOpts{optId})),
                    send_xmlcmd_udp('interaction-signal', ['i:' demoOpts{optId}], getfield(demo, demoOpts{optId}));
                else
                    send_xmlcmd_udp('interaction-signal', demoOpts{optId}, getfield(demo, demoOpts{optId}));
                end
                    pause(.005);
            end
             pause;
	% START PYFF AND START FEEDBACK
       send_xmlcmd_udp('interaction-signal', 'command', 'play'); 
       
      disp('Vorstellen der naechsten Kondition')
      pause;
      send_xmlcmd_udp('interaction-signal', 's:_feedback', 'p300_photobrowser','command','sendinit');
      pause(1)
    end
    
   %QUIT DEMO
   send_xmlcmd_udp('interaction-signal', 'command', 'quit'); 

%% EEG Aufnahme von offenen Augen 
disp('Open Eyes measurement, press <ENTER> to start')
pause;
disp('SURE? Open Eyes measurement will start, press <ENTER> to start')
pause;
bvr_startrecording(['eyesOpen_' VP_CODE], 'impedances', 1);

pause(90);

bvr_sendcommand('stoprecording');
 
fprintf('COMPLETED Open Eyes measurement, press <ENTER> to start the Closed-Eyes measurement \r')
pause;

%% EEG Aufnahme von geschlossenen Augen
disp('SURE? Closed Eyes measurement, press <ENTER> to start')
pause;
bvr_startrecording(['eyesClosed_' VP_CODE], 'impedances', 0);

pause(90);

bvr_sendcommand('stoprecording');
 
disp('COMPLETED Closed Eyes measurement!  press <ENTER> to continue to PhotoBrowser')
pause;

%%  Photobrowser Settings, Calibration

        fbsettings.screen_width = 1920;                                    % window width in pixels
		fbsettings.screen_height = 1200;                                    % window height in pixels
		fbsettings.screen_x = -1920;                                          % x position of left top corner of the window in pixels
		fbsettings.screen_y = 0;                                          % y position of left top corner of the window in pixels

	%	Variables: limits/counts
		fbsettings.num_blocks                              = 1;            
		fbsettings.num_trials                              = 5;
		fbsettings.num_subtrials_per_iteration             = 2;
	
	% 	Variables: durations and pauses (all times in MILLISECONDS)
		fbsettings.startup_sleep_duration 				    = 1000; 		% pause on startup to allow the classifier time to initialise. Set to 0 to disable. 
		fbsettings.cue_presentation_duration 				= 2000; 
		fbsettings.pre_cue_pause_duration 				    = 1000;
		fbsettings.post_cue_presentation_pause_duration 	= 1000;
		fbsettings.inter_trial_duration 					= 4000;
		fbsettings.stimulus_duration 						= 100;
		fbsettings.inter_stimulus_duration 				    = 100;
		fbsettings.inter_block_pause_duration 			    = 7000;
		fbsettings.inter_block_countdown_duration 		    = 3000;
		fbsettings.result_presentation_duration 			= 5000;

	% Variables: miscellaneous
		fbsettings.max_inter_score_duration 				= 1000;        % maximum time in milliseconds to allow between successive scores... 
                                                                           % being received 
        fbsettings.udp_markers_enable = true;                              % if True, activates ue,false,truetrue,true,true,false,trueP...
                                                                           % markers when send_parallel is called
		fbsettings.online_mode_enable = false;                             % if True, activates online mode
		fbsettings.row_col_eval_enable = true;                             % if True, activates row/column mode in the sequence generator
		fbsettings.display_selection_enable = false; %true                 % if True, activates the displaying of the selected object 
                                                                           % after each trial
		fbsettings.show_empty_image_frames_enable = false;                 % if True, empty slots in the grid will still contain the standard image frame graphic
		fbsettings.mask_sequence_enable = false;
                   
 % Erste Initialisierung von Pyff mit fbsettings
	    send_xmlcmd_udp('interaction-signal', 's:_feedback', 'p300_photobrowser','command','sendinit');
        pause(3);
        
 %send fbsettings to feedback!!
	    fbOpts = fieldnames(fbsettings);
      for optId = 1:length(fbOpts),
        if isnumeric(getfield(fbsettings, fbOpts{optId})),
            send_xmlcmd_udp('interaction-signal', ['i:' fbOpts{optId}], getfield(fbsettings, fbOpts{optId}));
        else
            send_xmlcmd_udp('interaction-signal', fbOpts{optId}, getfield(fbsettings, fbOpts{optId}));
        end
        pause(.005);
      end
      

    
%% Variables: switches between conditions (all booleans)
	    condition.flash_enable=false;
        condition.scaling_enable=false;
        condition.rotation_enable=false;
        condition.invert_enable=false;
        condition.mask_enable=false;
        condition_list = {'brightness', 'scale', 'rotation', 'invert', 'mask', 'combination'};
        
    % local Variabels
    
        nr_iterations        = [10,12,14];
        randNr_iterations    = zeros(30,6);
        orderConditions      = [ 1 2 3 4 5 6;...
                                 4 6 1 3 2 5;...
                                 3 1 5 2 6 4;...
                                 2 5 6 1 4 3;...
                                 6 4 2 5 3 1];   
            
        num_conditions       = 6; 
        Nblocks              = 5;
        i=0;
        condition.num_iterations_menge = zeros(1,5);
        
   % erste Spalte benutzen wir als index! der Rest sind zufaellige
   % Werte fuer die Anzahl der Iterationen(Wie oft ein Target gehighlightet wird)
       for row = 1:30
          randNr_iterations(row,1)=row;
         for col= 2:6
          randNr_iterations(row,col) = nr_iterations(round(1+ 2*rand()));
          end
       end
       
   % speicher von der Iterations anzahl der targets 
     dlmwrite([TODAY_DIR 'Iterationen.txt'], randNr_iterations, ' ');
        
       
%% Real Experiment 
   %pyff('startup');
   i = 0;
   for block = 1:Nblocks
         
      fprintf('Starting Block0%i , press <ENTER>\r',block)
      pause;
         
     for orderCondition_col = 1:num_conditions
           
          i = i+1;
      
        % findet heraus welche die aktuelle Kondition ist
           switch orderConditions(block,orderCondition_col);
                case 1
                    disp('Flash')
                    condition.flash_enable=true;
                    condition.scaling_enable=false;
                    condition.rotation_enable=false;
                    condition.invert_enable=false;
                    condition.mask_enable=false;

                case 2
                   disp('Scale')
                    condition.flash_enable=false;
                    condition.scaling_enable=true;
                    condition.rotation_enable=false;
                    condition.invert_enable=false;
                    condition.mask_enable=false;
                case 3
                   disp('Rotation')
                    condition.flash_enable=false;
                    condition.scaling_enable=false;
                    condition.rotation_enable=true;
                    condition.invert_enable=false;
                    condition.mask_enable=false;
                case 4
                   disp('Invert') 
                   condition.flash_enable=false;
                    condition.scaling_enable=false;
                    condition.rotation_enable=false;
                    condition.invert_enable=true;
                    condition.mask_enable=false;
               case 5
                    disp('Mask')
                    condition.flash_enable=false;
                    condition.scaling_enable=false;
                    condition.rotation_enable=false;
                    condition.invert_enable=false;
                    condition.mask_enable=true;
               case 6
                    disp('Combination')
                    condition.flash_enable=true;
                    condition.scaling_enable=true;
                    condition.rotation_enable=true;
                    condition.invert_enable=false;
                    condition.mask_enable=true;
               otherwise
                   disp('Unknown condition')
           end
            cond = char(condition_list_fname{orderConditions(block,orderCondition_col)})
           condition.num_iterations_menge = randNr_iterations(i,2:6);
                         
   
   fprintf('condition = %s,blocknr=%i,Spalte=%i\r',cond,block,orderCondition_col); 
   fprintf('iteration = %i\r',condition.num_iterations_menge);

	       
       % START RECORDING
               bvr_startrecording(['PhotoBrowser_Condition_' cond '_' VP_CODE], 'impedances', 0);
           
       
               
       % SEND Condition_struct
            cdOpts = fieldnames(condition); 
            
            for optId = 1:length(cdOpts),
                if isnumeric(getfield(condition, cdOpts{optId})),
                    send_xmlcmd_udp('interaction-signal', ['i:' cdOpts{optId}], getfield(condition, cdOpts{optId}));
                else
                    send_xmlcmd_udp('interaction-signal', cdOpts{optId}, getfield(condition, cdOpts{optId}));
                end
                    pause(.005);
            end
            

            
       % START PYFF AND START FEEDBACK
            send_xmlcmd_udp('interaction-signal', 'command', 'play'); 
            
            disp('press <ENTER> if finished')
            pause;
            
            disp('Are you SURE ? recording will be stopped, press <ENTER> to finish')
            pause;
            fprintf('TRUE randNr = %i\r',randNr_iterations(block,orderCondition_col));
            
        % STOP PYFF 
            send_xmlcmd_udp('interaction-signal', 'command', 'stop');
        % STOP RECORDING
            bvr_sendcommand('stoprecording');
            
            disp('COMPLETED Condition!  press <ENTER> to continue to the next condition')
            pause;  
      end 
      fprintf('Block COMPLETED\r')
   end
   
 %% END
   fprintf('EXPERTIMENT ENDED,THANK YOU FOR YOUR TIME!')
   %QUIT PYFF
   send_xmlcmd_udp('interaction-signal', 'command', 'quit'); 
   
   
   
%% sonderkondition: zoom mit 20%
fprintf('bitte den Zoomfaktor auf 20% setzen...')
pause;

orderConditions(1,:) = 2;
orderConditions(2,:) = 2;

condition_list_fname = {'brightness'  'scale_20perc' 'rotation' 'invert' 'mask' 'combination'}

fprintf('bitte nun die vorherige Zelle nochmal ausfueren')
pause
