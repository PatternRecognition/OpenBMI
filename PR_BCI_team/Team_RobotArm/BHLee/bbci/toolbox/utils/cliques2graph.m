function adj = cliques2graph(cli,clab)
% CLIQUES2GRAPH - Build graph adjacency matrix from list of cliques
%
% Converts a clique cell array into an adjacency matrix based on the channel
% positions in clab.
%
% Usage: adj = cliques2graph(cli,clab)
%
% IN     cli   - cell array of clique cell arrays. Cliques may be given
%                as numerical indices or as cell arrays of channel names.
%        clab  - cell array of channels.
% OUT    adj   - matrix of zeros and ones, giving adjacency of
%                channels. adj is a symmetric logical matrix.
%
% See also CHANIND
%

% kraulem 08/04
% Anton Schwaighofer, 09/04

error(nargchk(1, 2, nargin));

% Distinguish between cliques given as cell strings and numeric cliques
if iscellstr(cli{1}),
  if nargin<2,
    error('Input argument CLAB must be provided when cliques are cell strings');
  end
  % Store adjacency matrix as logical array to save memory
  adj = logical(zeros(length(clab)));
  for i = 1:length(cli),
    % Convert the clique, given as a cell array of strings, to numerical indices
    cliqueInd = chanind(clab, cli{i});
    % Clique is fully connected
    adj(cliqueInd,cliqueInd) = 1;
  end
else
  % The trivial case where cliques are numeric indices.
  % First need to find out about the number of graph nodes
  maxNode = -Inf;
  % This holds a copy of cli2, converted to a cell array of index vectors
  cli2 = cell(size(cli));
  for i = 1:length(cli),
    c = cli{i};
    if iscell(c),
      cli2{i} = [c{:}];
    else
      cli2{i} = c;
    end
    maxNode = max(max(cli2{i}), maxNode);
  end
  % Number of graph nodes must match length of channel labels, if given:
  if (nargin>=2) & (length(clab)~=maxNode),
    error('Clique set and length of channel labels does not match');
  end
  if isinf(maxNode),
    % Empty clique set:
    adj = logical([]);
  else
    % We have a valid clique set: Build adjacency matrix
    adj = logical(zeros(maxNode));
    for i = 1:length(cli),
      adj(cli2{i},cli2{i}) = 1;
    end
  end
end
