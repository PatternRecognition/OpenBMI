function out = val_ranksumtest(labels, out_cfy, siglevs)
%
% USAGE:
%    signif = val_ranksumtest(labels, out_cfy, siglevs)
%
% Performs a Wilcoxon rank sum test
%
% Asses whether the classification output distribution is
% significantly different for the two classes
%
% NOTE: This function is applicable to 2-class classification problems only
%
% Simon Scholler, 2011


p = ranksum(out_cfy(logical(labels(1,:))),out_cfy(logical(labels(2,:))));

if nargin>2
   out = p<siglevs;
else
   out = p;
end