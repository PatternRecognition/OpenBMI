function [Gred] = statutil_factors_cell2int(G)

% G is the orginal matrix of grouping variables:
% each row for is an observation; each column is a different grouping variable.
Gred = G;
for k = 1:size(G,2)
  groups = sort(unique(G(:,k)));
  
  for m = 1:length(groups)
    t = find(G(:,k) == groups(m));
    Gred(t,k) = m;
  end
end
