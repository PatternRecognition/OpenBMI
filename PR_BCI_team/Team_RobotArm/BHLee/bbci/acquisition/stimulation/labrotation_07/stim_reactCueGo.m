function stim_matthiasVisual(N, varargin)
%ODDBALL_VISUAL - Provides Stimuli for Visual Oddbal
%
%Synopsis:
% stim_oddballVisual(N, <OPT>)
% stim_oddballVisual(N, PERC_DEV, <OPT>)
%
%Arguments:
% N: Number of stimuli
% PERC_DEV: Percentage of deviants
% OPT: Struct or property/value list of optional properties:
%  'perc_dev': Scalar: percentage of deviants
%  'isi':      Scalar: iner-stimulus interval [ms]
%  'duration_cue': Scalar: duration for which the stimulus is shown [ms]
%  'require_response': Boolean: if true, a response stimulus is expected
%     within the ISI.
%  'response_markers': Cell array 1x2 of strings: description of response
%     markers expected for STD and DEV stimuli. Default: {'R 16','R  8'}.
%
%Triggers:
% 111: left no-go cue, lower delay
% 112: left no-go cue, random delay
% 113: left no-go cue, upper delay
% 121: right no-go cue, lower delay
% 122: right no-go cue, random delay
% 123: right no-go cue, upper delay
% 211: left go cue, lower delay
% 212: left go cue, random delay
% 213: left go cue, upper delay
% 221: right go cue, lower delay
% 222: right go cue, random delay
% 223: right go cue, upper delay
% 1: beginning of relaxationg period
% 2: beginning of main experiment (after countdown)
% 3: end of main experiment
% 4: end

% blanker@cs.tu-berlin.de

global VP_CODE

opt= propertylist2struct(varargin{:});
opt= set_defaults(opt, ...
                  'test', 0);

cue_sequence = ones(1,N);
cue_sequence(1:round(N*opt.perc_dev)) = 2;
cue_sequence = cue_sequence(randperm(N));

delay_type = ceil(rand(1,N)*3);
cue_delay = zeros(1,N);
for i = 1:N
  switch delay_type(i)
    case 1
      cue_delay(i) = opt.lower_delay;
    case 2
%      cue_delay(i) = opt.rnd_delay_mean + opt.rnd_delay_std * (rand(1)-0.5);
      cue_delay(i) = opt.range_delay(1) + rand(1)*diff(opt.range_delay);
    case 3
      cue_delay(i) = opt.upper_delay;
  end
end

if ~isempty(opt.bv_host),
  bvr_checkparport;
end

[h_msg, opt.handle_background]= stimutil_initMsg(opt);
set(h_msg, 'String',opt.msg_intro, 'Visible','on');
drawnow;
waitForSync;

if opt.test,
  fprintf('Warning: test option set true: EEG is not recorded!\n');
else
  if ~isempty(opt.filename),
    bvr_startrecording([opt.filename VP_CODE]);
  else
    warning('*NOT* recording: opt.filename is empty');
  end
  ppTrigger(1);
  waitForSync(opt.duration_intro);
end

set(opt.handle_cross, 'Visible','on');
set(h_msg, 'Visible','off');
drawnow;

if ~opt.test,
  pause(1);
  stimutil_countdown(opt.countdown, opt);
  ppTrigger(2);
end
pause(1);

waitForSync;
for i = 1:N
  
  disp(num2str(i))

  if cue_sequence(i)==1
    hnd = opt.nogo_cue_left;
    ppTrigger(100 + 10*cue_sequence(i) + delay_type(i));
  else
    hnd = opt.nogo_cue_right;
    ppTrigger(100 + 10*cue_sequence(i) + delay_type(i));
  end
  set(hnd, 'Visible','on');
  drawnow;

  waitForSync(cue_delay(i));
  
  if strcmp(opt.stim_type,'visual')
    set(hnd, 'Visible','off');
    if cue_sequence(i)==1
      hnd = opt.go_cue_left;
      ppTrigger(200 + 10*cue_sequence(i) + delay_type(i));
    else
      hnd = opt.go_cue_right;
      ppTrigger(200 + 10*cue_sequence(i) + delay_type(i));
    end
    set(hnd, 'Visible','on');
    drawnow;    
  elseif strcmp(opt.stim_type,'auditory')
    if cue_sequence(i)==1
      wavplay(opt.go_cue_left, opt.fs, 'async');
      ppTrigger(200 + 10*cue_sequence(i) + delay_type(i));
    else
      wavplay(opt.go_cue_right, opt.fs, 'async');
      ppTrigger(200 + 10*cue_sequence(i) + delay_type(i));
    end    
  end
  
  waitForSync(opt.duration_cue);
  %ppTrigger(100);
  set(hnd, 'Visible','off');
  drawnow;
  waitForSync(opt.intertrial(1) + rand*opt.intertrial(2));
end

ppTrigger(3);
set(opt.handle_cross, 'Visible','off');
drawnow;
pause(1);

ppTrigger(4);
pause(1);
if ~opt.test && ~isempty(opt.filename),
  bvr_sendcommand('stoprecording');
end

pause(2);
delete(h_msg);
