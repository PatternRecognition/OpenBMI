function [ out ] = opt_stackHistory(in, c )
%OPT_STACKHISTORY Summary of this function goes here
%   Detailed explanation goes here
if isstruct(in)
    c = strsplit(c,'\');
    if isfield(in,'stack')
        in.stack{end+1}=c{end};
    end
else
    return;
end
out=in;
end

