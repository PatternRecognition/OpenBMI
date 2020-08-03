function [S, setupfile]= getClassifierLog(subdir, varargin)
%GETCLASSIFIERLOG - Information of BBCI classifier setup of specific experiment
%
%Synopsis:
% [S, SETUP_FILENAME]= getClassifierLog(SUBDIR)
%
%Arguments:
% SUBDIR: String. Name of a subdirectory of EEG_RAW_DIR.
%
%Returns:
% S: struct with fields bbci, cls, feature, cont_proc, post_proc, marker_output
% SETUP_FILENAME: name of the chosen setup file

% Author(s): Benjamin Blankertz, Oct 2007

global EEG_RAW_DIR EEG_MAT_DIR

opt= propertylist2struct(varargin{:});
opt= set_defaults(opt, ...
                  'setup_prefix', '*_', ...
                  'setup_dir', EEG_RAW_DIR);
%                  'setup_prefix', 'bbci_classifier_');

%dd= dir([opt.setup_dir '/' subdir '/' opt.setup_prefix 'setup_*_log']);
dd= dir([opt.setup_dir '/' subdir '/' opt.setup_prefix 'setup_*.mat']);
if isempty(dd) & isequal(opt.setup_dir, EEG_RAW_DIR),
  [S, setupfile]= getClassifierLog(subdir, opt, 'setup_dir', EEG_MAT_DIR);
  return;
end

if isempty(dd),
  S= [];
  setupfile= [];
  return;
end

if length(dd)==1,
  iSetup= 1;
else
  dd= dir([opt.setup_dir '/' subdir '/' opt.setup_prefix 'setup_*_log']);
  if isempty(dd),
    error('multiple log files, but no log subfolder');
  end
  nLog= ones(1, length(dd));
  for ii= 1:length(dd),
    ds= dir([opt.setup_dir '/' subdir '/' dd(ii).name]);
    nLog(ii)= length(ds)-2;  %% two entries are always '.' and '..'
  end
  iLog= nLog>0;
  if sum(iLog)==1,
    iSetup= find(iLog);
  elseif sum(iLog)>1,
    [mm, iSetup]= max(nLog);
    msg= sprintf(['More than one log subdirectory is non-empty. ' ...
                  'Choosing %s as setup file.'], dd(iSetup).name(1:end-4));
    warning(msg);
  end
end
setupfile= dd(iSetup).name(1:end-4);

try,
  S= load([opt.setup_dir subdir '/' setupfile]);
catch,
  [S, setupfile]= getClassifierLog(subdir, opt, 'setup_dir', EEG_MAT_DIR);
end
