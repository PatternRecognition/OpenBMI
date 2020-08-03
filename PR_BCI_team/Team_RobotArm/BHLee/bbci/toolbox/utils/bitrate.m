function rate= bitrate(p, n)
%rate= bitrate(p, <n=2>)
%
% p: probability that desired selection will be selected
% n: number of choices

if ~exist('n','var'), n=2; end

izo= find(p==0 | p==1);
p([izo])= 0.5;
rate= log2(n) + p.*log2(p) + (1-p).*log2((1-p)/(n-1));
rate(izo)= log2(n);



