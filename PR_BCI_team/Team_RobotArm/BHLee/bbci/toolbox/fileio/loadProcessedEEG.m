function [cnt, mrk, mnt]= loadProcessedEEG(file, appendix, vars)
% loadProcessedEEG - Load pre-processed EEG data from .mat file
%
% Synopsis:
%   [cnt,mrk,mnt] = loadProcessedEEG(file, appendix,vars)
%   
% Arguments:
%   file: file name, or cell array of file names. File names/paths are
%       relative to EEG_MAT_DIR unless beginning with '/'. 
%       concatProcessedEEG is called when file is a cell array.
%   appendix: string, appendix to file name specifying preprocessing type
%   vars: cell array of string, each string is the name of a variable to
%       load. Default: {'cnt', 'mrk', 'mnt'}. 
%       Can be used, e.g., to only load the marker structure.
%   
% Returns:
%   cnt: contiuous EEG signals, see readGenericEEG
%   mrk: class markers, see makeClassMarker
%   mnt: electrode montrage, see setElectrodeMontage
%   
% Global variables:
%   EEG_MAT_DIR: Default path-prefix for files.
% 
% Examples:
%   loadProcessedEEG('mydata')
%      will attempt to load {EEG_MAT_DIR}/mydata.mat
%   loadProcessedEEG('/home/user/mydata', 'feedb')
%      will attempt to load /home/user/mydata_feedb.mat
%   loadProcessedEEG('H:/home/user/mydata', 'feedb')
%      does the job on a Windoze machine.
%
% See also: saveProcessedEEG, concatProcessedEEG
% 

% Author(s): bb, Anton Schwaighofer, Sep 2004
% $Id: loadProcessedEEG.m,v 1.10 2005/05/12 10:36:18 neuro_cvs Exp $

global EEG_MAT_DIR

if nargin<3,
  vars={'cnt','mrk','mnt'};
elseif ~iscell(vars),
  vars= {vars};
end
if nargin<2,
  appendix = '';
end

% Check for appendix. Remove any underscores and backspaces (\_ is used as an
% escape sequence in many title strings). I guess actually only leading
% underscores would need to be removed, along with file separator
% characters. For compatibility, I keep the original code here.
if ~isempty(appendix),
  appendix= appendix(find(~ismember(appendix, '_\')));
  appendix = ['_' appendix];
end
cnt= NaN; mrk= NaN; mnt= NaN;

%% This is a hack, to see whether the file is in the new data format.
%% If so, the new function eegfile_loadMatlab is used.
if iscell(file),
  fff= file{1};
else
  fff= file;
end
wstat= warning('off');
S= load(prefix_eegmatfile(fff, appendix), 'nfo');
warning(wstat);
if isfield(S, 'nfo'),
  vvv= cell(1, length(vars));
  [vvv{:}]= eegfile_loadMatlab(strcat(file, appendix), 'vars',vars);
  for ii= 1:length(vars),
    eval(sprintf('%s= vvv{%d};', vars{ii}, ii));
  end
  return;
end

%% Otherwise we continue with the old code:
if iscell(file),
  [cnt, mrk, mnt]= concatProcessedEEG(file, appendix, vars);
  if cnt.fs==0,
    cnt= NaN;
  end
  return;
end

fullName= prefix_eegmatfile(file, appendix);
if ~exist(fullName, 'file'),
  error(sprintf('file %s does not exist', fullName));
end

load(fullName, vars{:});

if isstruct(cnt),
  cnt.file= file;
end
