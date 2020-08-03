function out= bvr_startrecording(filebase, varargin)
%BVR_STARTRECORDING - Start Acquisition with BV RECORDER
%
% The actual filename for recording is composed of TODAY_DIR (global
% variable), a given basename, and optionally the VP_CODE (global variable).
% The a file of that name exists a counter is increased and the
% corresponding nummber is appended, until a new filename was created.
%
%Synopsis:
% FILENAME= bvr_startrecording(FILEBASE, <OPT>)
%
%Arguments:
% FILEBASE: basename for the EEG files.
% OPT: Struct or property/value list of optional properties:
%   'impedances': logical: start impedance measurement at the beginning
%   'append_VP_CODE': Append the VP_CODE (global variable) to file name
%
%Returns:
% FILENAME: Actually chosen filename
%
%Uses global variables
% TODAY_DIR and (if OPT.append_VP_CODE==true) VP_CODE

global TODAY_DIR VP_CODE

if isempty(TODAY_DIR),
  error('global TODAY_DIR is not set');
end

opt= propertylist2struct(varargin{:});
opt= set_defaults(opt, ...
                  'impedances', 1, ...
                  'append_VP_CODE', 0);
                
%% in case recording is still running, stop it
bvr_sendcommand('stoprecording');

if opt.append_VP_CODE,
  filebase= [filebase VP_CODE];
end

num= 1;
file= [TODAY_DIR '\' filebase];
[file '.eeg']
while exist([file '.eeg'], 'file'),
  num= num + 1;
  file= sprintf('%s%s%02d', TODAY_DIR, filebase, num);
end

if opt.impedances,
  bvr_sendcommand('startimprecording', [file '.eeg']);
else
  bvr_sendcommand('startrecording', [file '.eeg']);
end
fprintf('Saving to <%s>.\n', file);

if nargout>0,
  out= file;
end
