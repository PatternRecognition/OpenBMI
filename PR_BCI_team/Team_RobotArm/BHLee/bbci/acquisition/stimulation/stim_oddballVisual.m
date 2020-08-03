function stim_oddballVisual(N, varargin)
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
%   1: STD stimulus
%   2: DEV stimulus
% 100: Cue off
% 251: beginning of relaxationg period
% 252: beginning of main experiment (after countdown)
% 253: end of main experiment
% 254: end

% blanker@cs.tu-berlin.de

global VP_CODE

if length(varargin)>0 & isnumeric(varargin{1}),
  opt= propertylist2struct('perc_dev',varargin{1}, varargin{2:end});
else
  opt= propertylist2struct(varargin{:});
end

opt= set_defaults(opt, ...
                  'filename', 'oddball_visu', ...
                  'test', 0, ...
                  'perc_dev', 0.15, ...
                  'isi', 1500,...
                  'isi_jitter', 0, ...
                  'duration_cue', 500,...
                  'require_response', 1, ...
                  'response_markers', {'R 16', 'R  8'}, ...
                  'avoid_dev_repetitions', 1, ...
                  'background', 0.5*[1 1 1], ...
                  'countdown', 7, ...
                  'countdown_fontsize', 0.3, ...
                  'duration_intro', 7000, ...
                  'bv_host', 'localhost', ...
                  'msg_intro','Entspannen', ...
                  'msg_fin', 'fin', ...
                  'delete_obj', []);

if ~isfield(opt, 'cue_dev') | ~isfield(opt, 'cue_std'),
  error('opt must have fields ''cue_dev'' and ''cue_std''.');
end

cue_sequence= zeros(1,N);
devs_to_be_placed= round(N*opt.perc_dev);
if opt.avoid_dev_repetitions,
  ptr= 1;
  while ptr<N,
    ptr= ptr + 1;
    togo= N-ptr+1;
    prob= devs_to_be_placed/floor(togo/2);
    if rand<prob,
      cue_sequence(ptr)= 1;
      devs_to_be_placed= devs_to_be_placed - 1;
      ptr= ptr + 1;
    end
  end
else
  cue_sequence(1:devs_to_be_placed)= 1;
  cue_sequence= cue_sequence(randperm(N));
end

if ~opt.test & ~isempty(opt.bv_host),
  bvr_checkparport;
end

delete(opt.delete_obj(find(ishandle(opt.delete_obj))));
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
pause(1);

if opt.require_response,
  response= zeros(N, 1);
  correct= NaN*zeros(N, 1);
  state= acquire_bv(1000, opt.bv_host);
end
waitForSync;
for i= 1:N,
  if cue_sequence(i),
    ci= ceil(rand*length(opt.cue_dev));
    hnd= opt.cue_dev(ci);
  else
    ci= ceil(rand*length(opt.cue_std));
    hnd= opt.cue_std(ci);
  end
  ppTrigger(1+cue_sequence(i));
  set(hnd, 'Visible','on');
  drawnow;
  trial_duration= opt.isi + rand*opt.isi_jitter;
  if opt.require_response,
    t0= clock;
    resp= [];
    [dmy]= acquire_bv(state);  %% clear the queue
    while isempty(resp) & 1000*etime(clock,t0)<trial_duration-50,
      [dmy,bn,mp,mt,md]= acquire_bv(state);
      for mm= 1:length(mt),
        resp= strmatch(mt{mm}, opt.response_markers);
        if ~isempty(resp),
          continue;
        end
      end
      pause(0.001);  %% this is to allow breaks
    end
    response(i)= etime(clock,t0);
    if ~isempty(resp),
      correct(i)= (resp==2-cue_sequence(i));
    end
    fprintf('%d:%d  (%d missed)\n', sum(correct==1), sum(correct==0), ...
      sum(isnan(correct(1:i))));
  end
  waitForSync(opt.duration_cue);
  ppTrigger(100);
  set(hnd, 'Visible','off');
  drawnow;
  waitForSync(trial_duration - opt.duration_cue);
end

ppTrigger(253);
set(opt.handle_cross, 'Visible','off');
drawnow;
pause(1);

if opt.require_response,
  iGood= find(correct==1);
  iBad= find(correct==0);
  msg= sprintf('%d hits  :  %d errors  (%d missed)', ...
               length(iGood), length(iBad), sum(isnan(correct)));
  set(h_msg, 'String',msg, ...
                    'Visible','on', ...
                    'FontSize', 0.5*get(h_msg,'FontSize'));
  fprintf('%s\n', msg);
  fprintf('|  average response time:  ');
  if ~isempty(iGood),
    fprintf('%.2f s on hits  | ', mean(response(iGood)));
  end
  if ~isempty(iBad),
    fprintf('%.2f s on errors  |', mean(response(iBad)));
  end
  fprintf('\n');
else
  set(h_msg, 'String',opt.msg_fin);
end

ppTrigger(254);
pause(1);
if ~opt.test & ~isempty(opt.filename),
  bvr_sendcommand('stoprecording');
end

pause(5);
delete(h_msg);
