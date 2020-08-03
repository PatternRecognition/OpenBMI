function [separatorList, separators2node, separatorMatrix] = cliques2separators(cliqueSet, verbosity)
% CLIQUES2SEPARATORS - Compute all separators from a given set of cliques
%
%   [SEPARATORLIST, SEPARATORS2NODE, SEPARATORMATRIX] = ...
%      CLIQUES2SEPARATORS(CLIQUESET, VERBOSITY)
%   From a given set of cliques (stored as a cell array, CLIQUESET{i} is
%   the set of variables falling into clique i) compute all
%   separators. Separators are again returned as cell arrays, yet in two
%   (equivalent) formats:
%   SEPARATORLIST{k} is the set of variables in the k.th separator. Mind
%   that there may exist several separators sharing exactly the same
%   variables.
%   SEPARATORMATRIX{i,j} is the set of variables connecting cliques
%   CLIQUESET{i} andCLIQUESET{j}
%
%   SEPARATORS2NODE is a cell array that contains in SEPARATORS2NODE{n}
%   the list of separators that contain variable/node n. The list of
%   separators is given as a list of indices in SEPARATORLIST.
%   
%   See also CLIQUES_TO_JTREE
%

% 
% Copyright (c) by Anton Schwaighofer (2002)
% $Revision: 1.1 $ $Date: 2004/08/19 12:12:36 $
% mailto:anton.schwaighofer@gmx.net
% 

error(nargchk(1, 2, nargin));
if nargin<2,
  verbosity = 1;
end

separatorList = cell(0);
separatorMatrix = cell([length(cliqueSet) length(cliqueSet)]);

maxnode = max([cliqueSet{:}]);
JT = triu(cliques_to_jtree(cliqueSet, ones([1 maxnode])));
JTind = find(JT);
for k = JTind',
  [i,j] = ind2sub(size(JT), k);
  sep = intersect2(cliqueSet{i},cliqueSet{j});
  if ~isempty(sep),
    separatorList{end+1} = sep;
    separatorMatrix{i,j} = sep;
    if verbosity>0,
      fprintf('Separator %i (between cliques %i and %i): %s\n', ...
              length(separatorList), i, j, dispset(sep));
    end
  end
end

separators2node = cell([1 maxnode]);
for i = 1:length(separatorList),
  for s = separatorList{i},
    separators2node{s} = union(separators2node{s}, i);
  end
end
