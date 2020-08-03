function classnames = getClassnames(files,varargin)
%getClassnames read out of all given EEG files all used classes.
%
% usage: 
%   classnames = getClassnames(files,...);
% 
% input:
%   files   -   a cell array of filenames (must be loadable with
%               loadProcessedEEG),alternatively a series of file
%               inputs
%
% output:
%   classnames - a cell array of used classnames (concatenation of
%                all classnames without using once twice)
%
% Guido Dornhege, 19/09/2003

if ~iscell(files)
  files = {files};
end

files = {files{:},varargin{:}};

classnames = {};

for i = 1:length(files);
  [dum,mrk] = loadProcessedEEG(files{i},[],'mrk');
  for j = 1:length(mrk.className)
    classi = mrk.className{j};
    if sum(strcmp(classnames,classi))==0
      classnames = cat(2,classnames,{classi});
    end
  end
end

  