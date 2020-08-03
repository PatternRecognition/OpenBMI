function fullName= prefix_eegmatfile(file, appendix)
%fullName= prefix_eegmatfile(file, <appendix>)
%
% This function prepends the global string EEG_MAT_DIR to the given
% file name, unless it contains an absolute path.

if nargin==1,
  appendix= '';
end

% Distinguish between absolute and relative paths: 
% For Unix systems, absolute paths start with \
if isunix & (file(1)==filesep),
  prefix = '';
elseif ispc & (file(2)==':'),
  % For Windoze, identify absolute paths by the : (H:\some\path)
  prefix = '';
else
  % Does not seem to be an absolute path: Path is relative to EEG_MAT_DIR
  global EEG_MAT_DIR
  prefix= EEG_MAT_DIR;
end
% Dissect file name, rebuild full path from prefix and the given appendix
[pathstr, name, ext, versn] = fileparts(file);
% Default extension: .mat
if isempty(ext),
  ext = '.mat';
end
fullName = fullfile(prefix, pathstr, [name appendix ext versn]);
