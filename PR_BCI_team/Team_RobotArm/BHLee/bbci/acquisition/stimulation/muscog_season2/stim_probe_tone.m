function stim_probe_tone(order,sounds_key,varargin)
% stim_probe_tone - provides Stimuli for probe tone experiments
% before running stim_probe_tone setup_probe_tone 
%Synopsis:
% stim_probe_tone(order,sounds_key,<OPT>)
% 
%Arguments:
% 
% order:        struct with fields
%               .probe_tones:  [1 nTrials]- vector specifying the intervals
%               in semitones for each change of key.  
%               .sequence: sequence of keys that is the result of changing
%               keys according to the
%               probe_tone-vector starting at a randomly initialized
%               starting point.
% sounds_key:   {1,12} cell array holding the sounds (12 pitches)
%Triggers:
%   
% 251: beginning of relaxationg period
% 252: beginning of main experiment (after countdown)
% 253: end of main experiment
% 254: end
% 1-12      : key e.g. 1 for C major, 2 for C# major, 3 for C# Major, ...
% 
% 
% blanker@cs.tu-berlin.de

global VP_CODE BCI_DIR

if length(varargin)>0 & isnumeric(varargin{1}),
  opt= propertylist2struct('perc_dev',varargin{1}, varargin{2:end});
else
  opt= propertylist2struct(varargin{:});
end

opt= set_defaults(opt, ...
                  'filename', 'probe_tone', ...
                  'test', 0, ...
                  'require_response', 1, ...
                  'response_markers', {'R 16', 'R  8'}, ...
                  'background', 0.5*[1 1 1], ...
                  'break_duration',15,...
                  'countdown', 7, ...
                  'countdown_fontsize', 0.3, ...
                  'duration_intro', 7000, ...
                  'bv_host', 'localhost', ...
                  'msg_intro','Entspannen', ...
                  'msg_fin', 'Ende');
                  
if ~isempty(opt.bv_host),
  bvr_checkparport;
end

[h_msg, opt.handle_background]= stimutil_initMsg;
set(h_msg, 'String',opt.msg_intro, 'Visible','on');
drawnow;
waitForSync;

if opt.test,
  fprintf('Warning: test option set true: EEG is not recorded!\n');
else
  if ~isempty(opt.filename),
    bvr_startrecording([opt.filename VP_CODE]);
  else
    warning('!*NOT* recording: opt.filename is empty');
  end
  ppTrigger(251);
  waitForSync(opt.duration_intro);
end

if ~isfield(opt, 'handle_cross') | isempty(opt.handle_cross),
  opt.handle_cross= stimutil_fixationCross(opt);
else
  set(opt.handle_cross, 'Visible','on');
end

set(h_msg, 'Visible','off');

if ~opt.test,
  pause(1);
  stimutil_countdown(opt.countdown, opt);
  ppTrigger(252);
end
%Init:
%if (opt.require_response==1)
%  disp('init')
%  state= acquire_bv(1000, opt.bv_host);
%  opt.state=state;
%opt.response_markers= {'R 53', 'R 54', 'R 55', 'R 56', 'R 57', 'R 58', 'R 59'}; % Marker für '1' bis '7': Erlaubte Eingaben
%end

%%%%%%%%%%%%%%%%%%beginning of main experiment
pause(1);


set(opt.handle_cross, 'Visible','on');    
probe_tone_exp(order,sounds_key,opt)
set(opt.handle_cross, 'Visible','off'); 
    
set(h_msg, 'String',opt.msg_fin);
set(h_msg, 'Visible','on');


ppTrigger(254);
pause(1);
if ~opt.test & ~isempty(opt.filename),
  bvr_sendcommand('stoprecording');
end

pause(5);
delete(h_msg);
