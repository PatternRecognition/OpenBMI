function [p, iDel]= path_exclude(p, pattern)
%PATH_EXCLUDE - Exclude subfolders from a path string
%
%Synopsis:
% NEW_PATH= path_exclude(OLD_PATH, PATTERN)
%
%Arguments:
% OLD_PATH: String defining a path as returned by genpath.
% PATTERN: String specifies the subfolders that should be excluded as
%    a regular expression, see regexp.
%
%Returns:
% NEW_PATH: Path string with specified subfolders excluded.
%
%Example:
% addpath(path_exclude(genpath(SOME_SVN_DIR), '.*/\.svn.*')); %% Unix
% addpath(path_exclude(genpath(SOME_SVN_DIR), '.*\\.svn.*')); %% Windows

% blanker@cs.tu.berlin.de

if isunix,
  inter= [0, find(p==':')];
else
  inter= [0, find(p==';')];
end
iDel= [];
for sp= 1:length(inter)-1,
  idx= inter(sp)+1:inter(sp+1);
  str= p(idx(1:end-1));
  if ~isempty(regexp(str, pattern)),
    iDel= cat(2, iDel, idx);
  end
end
p(iDel)= [];
