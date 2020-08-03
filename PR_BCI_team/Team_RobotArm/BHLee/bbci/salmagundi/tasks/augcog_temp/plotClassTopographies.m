function h= plotClassTopographies(epo, mnt, ival, varargin)
%h= plotClassTopographies(epo, mnt, ival, opt)
%
% opt is passed to plotScalpPattern
% h is vector of axes' handles

if isfield(epo, 'xUnit') & isequal(epo.xUnit, 'Hz'),
  iv= min(find(epo.t>=ival(1))):max(find(epo.t<=ival(2)));
  fmt= '%.1f';
  unit= 'Hz';
else
  iv= getIvalIndices(ival, epo);
  fmt= '%d';
  unit= 'ms';
end
nClasses= size(epo.y, 1);
for ip= 1:nClasses,
  subplot(1, nClasses, ip);
  clInd= find(epo.y(ip,:));  
  w= mean(mean(epo.x(iv, :, clInd), 1), 3);
  h(ip)= plotScalpPattern(mnt, w, varargin{:});

  if isfield(epo, 'className'),
    tit= [epo.className{ip} '  '];
  else
    tit= [];
  end
  tit= sprintf(['%s[' fmt ' ' fmt '] ' unit], tit, epo.t(iv([1 end])));
  title(tit);
end

if isfield(epo, 'title'),
  addTitle(untex(epo.title), 1, 0);
end
