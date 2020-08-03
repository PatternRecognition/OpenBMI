function stim_probe_tone(varargin)
% stim_probe_tone - Provides Stimuli for probe tone experiments
% before running stim_probe_tone setup_probe_tone_exp and
% init_probe_tone_exp have to be run.
%Synopsis:
% stim_probe_tone(<OPT>)
% 
%Arguments:
% 
% OPT: Struct or property/value list of optional properties:
%  
%  'require_response': Boolean: if true, a response stimulus is expected
%     within the ISI.
%  'response_markers': Cell array 1x2 of strings: description of response
%     markers expected for  stimuli. Default: {'R 101' - 'R  107'}.
%
%Triggers:
%   
% 251: beginning of relaxationg period
% 252: beginning of main experiment (after countdown)
% 253: end of main experiment
% 254: end

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
                  'msg_fin', 'Ende', ...
                  'sequences_until_break',36, ... 
                  'block',1, ...
                  'blocksize',72);

%which block of experiment?               
block=opt.block;
fn= [BCI_DIR 'stimulation/praktikum07_bci02/' VP_CODE '_sequences_matrices'];
load (fn);

if (mod(size(sequence_matrices{block},2),opt.sequences_until_break)~=0)
    fprintf('opt.sequences_until_break is not a divisor of opt.blocksize. Last block has less than opt.sequences_until_break sequences.');
end

for k=1:(round(size(sequence_matrices{block},2)/opt.sequences_until_break))
    if (k==(round(size(sequence_matrices{block},2)/opt.sequences_until_break)))
         seqs{k}=(sequence_matrices{block}(:,(k-1)*opt.sequences_until_break+1:end));
    else     
    seqs{k}=sequence_matrices{block}(:,(k-1)*opt.sequences_until_break+1:k*opt.sequences_until_break);
    end
end
fprintf('Using sequence block %d: ', block);
fprintf('%d, ', seqs{k}(1,1:10));
fprintf('\n');

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
if (opt.require_response==1)
  disp('init')
%  state= acquire_bv(1000, opt.bv_host);
%  opt.state=state;
%opt.response_markers= {'R 53', 'R 54', 'R 55', 'R 56', 'R 57', 'R 58', 'R 59'}; % Marker für '1' bis '7': Erlaubte Eingaben
end

%%%%%%%%%%%%%%%%%%beginning of main exp.
pause(1);

for k=1:length(seqs)
set(opt.handle_cross, 'Visible','on');    

[tmp_opt exp_matrix]=probe_tone_exp(opt,'order',seqs{k});
set(opt.handle_cross, 'Visible','off'); 
    if (k<length(seqs))
        stimutil_break(opt);
    end
end
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
