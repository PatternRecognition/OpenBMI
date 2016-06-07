function [ out ] = opt_cnt2smt( dat )
% opt_cnt2smt :
%  This function reshapes the 2-dimensional data into 3-dimensional data.
% 
% Example:
%    out = opt_cnt2smt(dat,{'dimInfo',[500 100 64]})
% 
if ~isfield(out, dimInfo)
    warning('OpenBMI: The "dimInfo" is missing: dat.dimInfo');
else
    [nD nTr nCH]=out.dimInfo;
    dat=reshape(dat, [nD nTr nCH]);
    out.x=dat;
end

end

