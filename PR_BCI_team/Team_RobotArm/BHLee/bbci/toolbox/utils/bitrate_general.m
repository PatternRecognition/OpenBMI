function rate= bitrate_general(p, n)
%rate= bitrate(p, <n=2>)
%
% p: probability that desired selection will be selected
% n: number of choices

if size(p,1)>2
  error('only two class implemented so far');
end
pp = p./repmat(sum(p,1),[size(p,1),1]);
p = fminbnd('bitrate_opt',0.001,0.999,optimset,pp);
rate = -bitrate_opt(p,pp);



