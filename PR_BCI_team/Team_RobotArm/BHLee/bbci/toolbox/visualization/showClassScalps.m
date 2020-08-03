function h= showClassScalps(epo, mnt, ival, varargin)
%h= showClassScalps(epo, mnt, ival, params)
%
% params are passed to showScalpPattern from COL_AX arg on
% h is vector of axes' handles

if isfield(epo, 'xUnit') & isequal(epo.xUnit, 'Hz'),
  iv= min(find(epo.t>=ival(1))):max(find(epo.t<=ival(2)));
else
  iv= getIvalIndices(ival, epo);
end
nClasses= size(epo.y, 1);
for ip= 1:nClasses,
  subplot(1, nClasses, ip);
  clInd= find(epo.y(ip,:));  
  w= mean(mean(epo.x(iv, :, clInd), 1), 3);
  h(ip)= showScalpPattern(mnt, w, 0, 'horiz', varargin{:});

  if isfield(epo, 'className'),
    tit= [epo.className{ip} '  '];
  else
    tit= [];
  end
  tit= sprintf('%s[%.1f %.1f]', tit, epo.t(iv([1 end])));
  title(tit);
end

if isfield(epo, 'title'),
  addTitle(untex(epo.title), 1, 0);
end
