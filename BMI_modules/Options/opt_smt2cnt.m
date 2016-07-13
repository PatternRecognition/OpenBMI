function [ out datInfo] = opt_smt2cnt(Dat)
% opt_smt2cnt: 


if isstruct(Dat)
    out=Dat;
    if isfield(Dat, 'x')
        dat=Dat.x;
        if ndims(dat)==3  % smt2cnt
            [nD nTr nCH]=size(dat);
            dat=reshape(dat, [nD*nTr nCH]);
            out.x=dat;
            out.dimInfo=[nD nTr nCH];
        elseif ndims(dat)==2 %cnt2smt
            warning('OpenBMI: check the dimensions of Dat.x(DATA X TRIAL X CH)');
        end
    else
        warning('OpenBMI: check the field of x from input data: Dat.x');
    end
else 
    dat=Dat;
    if ndims(dat)==3  % smt2cnt
        [nD nTr nCH]=size(dat);
        dat=reshape(dat, [nD*nTr nCH]);
        out=dat;
        datInfo=[nD nTr nCH];
    else
        warning('OpenBMI: check the dimensions of Dat.x, 2(DATA X CH) or 3(DATA X TRIAL X CH) are acceptable');
    end    
end

