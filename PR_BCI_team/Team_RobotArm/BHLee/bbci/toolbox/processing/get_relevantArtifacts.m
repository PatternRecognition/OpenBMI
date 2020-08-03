function iv = get_relevantArtifacts(mrk,varargin)
% mrk is the usual artifacts field
% varargin is a number of arguments, where each argument is a
% string with the name of the artifact or a cell  array with more
% than on strings for artifacts you wants to combine as one
% artifact, where the first string represent the new name of
% the artifact 
% you get back a structure with iv.artifacts are the name of the
% artifacts and iv.int is a cell array with the intervals for each
% artifacts in a number artifacts*2 matrice
% if varargin doesn't exist, all artifacts are given back

if ~exist('varargin') | isempty(varargin)
  varargin = mrk.artifacts;
end

iv.artifacts = cell(length(varargin),1);
iv.int = cell(length(varargin),1);
for i =1:length(varargin)
  s = varargin{i};
  if ~iscell(s)
    s={s};
  end    
  iv.artifacts{i,1} = s{1};
  for j=1:length(s)
    c = find(strcmp(s{j},mrk.artifacts));
    iv.int{i} = cat(1,iv.int{i},mrk.int(find(mrk.y==c),:));
  end
end
  
