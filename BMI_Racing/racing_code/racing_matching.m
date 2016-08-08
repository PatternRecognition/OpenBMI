function [acc]=racing_matching(cf_out,true_lb)
% Example:
%   [acc]=racing_matching(CF_OUT2',TRUE_LABEL);
% CF_OUT2   : 808x1 double
% TRUE_LABEL: 1x808 double

match=cf_out==true_lb;
% acc=sum(match(find(true_lb)))/length(find(true_lb));
acc=sum(match)/length(match);