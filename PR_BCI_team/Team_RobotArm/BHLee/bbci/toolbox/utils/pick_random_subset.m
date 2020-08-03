function out= pick_random_subset(x, n)
%out= pick_random_subset(x, n)

nx= length(x);
idx= randperm(nx);
out= x(idx(1:min(n,nx)));
