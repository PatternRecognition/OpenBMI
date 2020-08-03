function cli = graph2cliques(adj)
% function cli = graph2cliques(adj)
%
% IN     adj - adjacency matrix (zeros and ones)
% OUT    cli - cell array with cliques.
%
% algorithm by Bron, Kerbosch 1971.

% kraulem 08/04
adj = logical(adj);
[comp_sub,candidates,no,cli] = extension_operation([], 1:size(adj, ...
                                                  1),[],adj,{});
return

function [comp_sub,candidates,no,cli]= ...
    extension_operation(comp_sub,candidates,no,adj,cli)
if isempty(candidates)
  if isempty(no)
    % otherwise: comp_sub is contained in a larger maximal subtree.
    cli{end+1} = comp_sub;
  end
  return
end
for i = 1:length(candidates)
  % 1. Selection of a candidate
    candidate = candidates(i);
    % 2. Adding the selected candidate to compsub
    comp_sub(end+1) = candidate;  
    % 3. Creating new sets candidates and no from the old sets...
    new_candidates = candidates(i+1:end);
    new_candidates = new_candidates(adj(candidate,candidates(i+1:end)));
    new_no = no(adj(candidate,no));
    % 4. extension operation
    [new_comp_sub,new_candidates,new_no,cli] = ...
        extension_operation(comp_sub,new_candidates,new_no,adj,cli);
    % 5. Removal of the selected candidate from compsub and addition to no.
    no(end+1) = comp_sub(end);
    comp_sub(end) = [];
end
return