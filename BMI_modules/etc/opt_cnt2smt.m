function [ out ] = opt_cnt2smt( Dat )
%OPT_CNT2SMT Summary of this function goes here
%   Detailed explanation goes here
if ~isfield(out, dimInfo)
    warning('OpenBMI: The "dimInfo" is missing: dat.dimInfo');
else
    [nD nTr nCH]=out.dimInfo;
    dat=reshape(dat, [nD nTr nCH]);
    out.x=dat;
end

end

