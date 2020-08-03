function [cue_seq twitch_seq]= stim_sssepBCI(N, opt);
%stim_sssepBCI - Stimulus Presentation for SSSEP stimuli
%
%Synopsis:
% SEQ= stim_oddballAuditory(N, <OPT>)
% SEQ= stim_oddballAuditory(N, PERC_DEV, <OPT>)
%
%Arguments:
% N: Number of stimuli
% PERC_DEV: Percentage of deviants
% OPT: Struct or property/value list of optional properties:
% 'perc_dev': Scalar: percentage of deviants
% 'iti':      Scalar: inter-trial interval [ms]

% 'response_markers': Cell array 1x2 of strings: description of response
%    markers expected for STD and DEV stimuli. Default: {'R 16','R  8'}.
% 'sound_files', Cell array of strings: file name of WAV file for
%    standard and deviant stimuli.
% 'msg_vpos': Scalar. Vertical position of message text object. Default: 0.57.
% 'msg_spec': Cell array. Text object specifications for message text object.
%   Default: {'FontSize',0.1, 'FontWeight','bold', 'Color',[.9 0 0]})
%
%Triggers:
%   1: left stimulus
%   2: right stimulus
% 251: beginning of relaxing period
% 200: beginning of trial
% 201: end of trial
% 252: beginning of main experiment ue(after countdown)
% 253: end of main experiment
% 254: end

%GLOBZ: BCI_DIR, VP_CODE

global BCI_DIR VP_CODE SOUND_DIR
% if length(varargin)>0 & isnumeric(varargin{1}),
%   opt= propertylist2struct('perc_dev',varargin{1}, varargin{2:end});
% else
%   opt= propertylist2struct(varargin{:});
% end
% sssep_BCI_27_new_lev8_dif
opt= set_defaults(opt, ...
                  'filename', 'sssep_bci_2729_lev8', ...
                  'test', 0, ...
                  'perc_dev', 0.5, ...
                  'iti', 2000,...
                  'iti_jitter', 1000, ...
                  'bv_host', 'localhost', ...
                  'countdown', 7, ...
                  'speech_dir', [SOUND_DIR 'english'], ...
                  'fs', 44100, ...
                  'msg_intro','Eyes open and relax', ...
                  'msg_fin','End', ...
                  'cue_dir',[SOUND_DIR 'cue_sssep'], ...
                  'cue_right','25', ...
                  'cue_left','27', ...
                  'cross_spec', {'Color',0*[1 1 1], 'LineWidth',4}, ...
                  'cross_size', 0.1, ...
                  'cross_vpos', 0.1, ...
                  'arrow_width', 0.15, ...
                  'arrow_size', 0.2, ...
                  'run',3, ...
                  'break_countdown',7, ...
                  'break_duration',15, ...
                  'break_msg', 'Short break for %d s', ...
                  'break_markers', [249 250]);

if ~isfield(opt, 'cue_left') | ~isfield(opt, 'cue_right'),
  error('opt must have fields ''cue_left'' and ''cue_right''.');
end

if ~isempty(opt.cue_left),
    signal_cue_right(:,1) = ...
      wavread([opt.cue_dir '/cue_' opt.cue_left '.wav']);
end  
if ~isempty(opt.cue_right),
    signal_cue_right(:,2) = ...
      wavread([opt.cue_dir '/cue_target_' opt.cue_right '.wav']);
end  

if ~isempty(opt.cue_left),
    signal_cue_left(:,1) = ...
      wavread([opt.cue_dir '/cue_target_' opt.cue_left '.wav']);
end  
if ~isempty(opt.cue_right),
    signal_cue_left(:,2) = ...
      wavread([opt.cue_dir '/cue_' opt.cue_right '.wav']);
end  

if ~isempty(opt.cue_left),
    signal_cue(:,1) = ...
      wavread([opt.cue_dir '/cue_' opt.cue_left '.wav']);
end  
if ~isempty(opt.cue_right),
    signal_cue(:,2) = ...
      wavread([opt.cue_dir '/cue_' opt.cue_right '.wav']);
end  

% if ~isempty(opt.bv_host),
%   bvr_checkparport;
% end
 

if opt.test,
  fprintf('Warning: test option set true: EEG is not recorded!\n');
else
  if ~isempty(opt.filename),
    bvr_startrecording([opt.filename VP_CODE]);
  else
    warning('!*NOT* recording: opt.filename is empty');
  end
waitForSync;
ppTrigger(251);  

opt.handle_background= stimutil_initFigure(opt,'position',[-1919 5 1920 1210]);%'
h_msg= stimutil_initMsg(opt);
set(h_msg, 'String',opt.msg_intro, 'Visible','on');
drawnow;

pause(3)
  
  for ii= opt.countdown:-1:1,
    set(h_msg, 'String',sprintf('Start in %d s', ii)); 
    drawnow;
    pause(1);
  end

  pause(1);
  set(h_msg, 'String',' ');
  ppTrigger(252);
  pause(1);
end

waitForSync;

stim.cue= struct('directions', {'left','right'});

h_arrow= stimutil_cueArrows({stim.cue.directions}); 


for ii=1:opt.run

pro = roundto(rand,0.1);
cue_sequence= zeros(1,N);
cue_sequence(1:round(N*opt.perc_dev))= 1;
cue_sequence= cue_sequence(randperm(N));
cue_seq(:,ii) = cue_sequence;    

twitch_sequence= zeros(1,N);
twitch_sequence(1:round(N*pro))= 1;
twitch_sequence= twitch_sequence(randperm(N));
twitch_seq(:,ii) = twitch_sequence;

for i= 1:N,
    ppTrigger(200);
        

        h_msg= stimutil_fixationCross(opt);
        set(h_msg, 'Visible','on');
        drawnow;
        
        waitForSync(2000);      
           
    if cue_sequence(i)==1,
      
        ppTrigger(1);
        
      
     if twitch_sequence(i)==1,
        wavplay(signal_cue_left, opt.fs, 'async');
        waitForSync(1000)
        ppTrigger(11);
        set(h_msg, 'Visible','off');
        set(h_arrow(1), 'Visible','on')
        drawnow;
     else
        wavplay(signal_cue, opt.fs, 'async');
        waitForSync(1000)
        ppTrigger(12);
        set(h_msg, 'Visible','off');
        set(h_arrow(1), 'Visible','on')
        drawnow;
     end
     waitForSync(3200);
     set(h_arrow(1), 'Visible','off');
     drawnow;
   else
     
     ppTrigger(2);   
         
        
     if twitch_sequence(i)==1,
        wavplay(signal_cue_right, opt.fs, 'async');
        waitForSync(1000)
        ppTrigger(22);
        set(h_msg, 'Visible','off');
        set(h_arrow(2), 'Visible','on')
        drawnow;
     else
        wavplay(signal_cue, opt.fs, 'async');
        waitForSync(1000)
        ppTrigger(23);
        set(h_msg, 'Visible','off');
        set(h_arrow(2), 'Visible','on')
        drawnow;
     end  
     waitForSync(3200);
     set(h_arrow(2), 'Visible','off');
     drawnow;
    end
    
   trial_duration= opt.iti + rand*opt.iti_jitter;
   ppTrigger(201);
   waitForSync(trial_duration);
end

    if ii < opt.run,
    
        stimutil_break(opt);
    else
        h_msg= stimutil_initMsg(opt);
        set(h_msg, 'String',opt.msg_fin, 'Visible','on');
        drawnow;

    end
end

ppTrigger(253);
pause(1);

ppTrigger(254);
pause(1);
if ~opt.test & ~isempty(opt.filename),
  bvr_sendcommand('stoprecording');
end
% 
% pause(5);
% delete(h_msg);
