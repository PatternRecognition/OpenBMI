function erg = rsqu(q, r)
% Description:  
%   rsqu(r, q) computes the r2-value for two one-dimensional distributions 
%   given by the vectors q and r
%
% Example code:
%  [erg] = rsqu(q , r)
%
% Input:
%   q: Data structrue (ex) Epoched data structure
%   r: Data structrue (ex) Epoched data structure
%
% Options:
%
% Return:
%    erg:  r2-values
%
% See also:
%
% Reference:
%           G. Schalk, D.J. McFarland, T. Hinterberger, N. Birbaumer, and
%           J. R. Wolpaw,"BCI2000: A General-Purpose Brain-Computer
%           Interface (BCI) System, IEEE Transactions on Biomedical
%           Engineering, Vol. 51, No. 6, 2004, pp.1034-1043.

%         We used BCI2000 open source toolbox code related in r-square value (rsqu.m)  
%
%  
% Ji Hoon, Jeong
% jh_jeong@korea.ac.kr

%%


q=double(q);
r=double(r);

sum1=sum(q);
sum2=sum(r);
n1=length(q);
n2=length(r);
sumsqu1=sum(q.*q);
sumsqu2=sum(r.*r);

G=((sum1+sum2)^2)/(n1+n2);

erg=(sum1^2/n1+sum2^2/n2-G)/(sumsqu1+sumsqu2-G);
end
