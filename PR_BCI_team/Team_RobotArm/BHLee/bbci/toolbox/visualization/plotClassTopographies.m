function [h,h_cb]= plotClassTopographies(epo, mnt, ival, varargin)
%h= plotClassTopographies(epo, mnt, ival, opt)
%
% opt is passed to plotScalpPattern, epo.arrangement is used here
% h is vector of axes' handles

opt= propertylist2struct(varargin{:});
opt= set_defaults(opt, ...
                  'unifiedCLim', 1, ...
                  'show_title', 1, ...
                  'titleAppendix', '', ...
                  'contour', 0, ...
                  'ax', []);

if isfield(epo, 'xUnit') & isequal(epo.xUnit, 'Hz'),
  iv= min(find(epo.t>=ival(1))):max(find(epo.t<=ival(2)));
%  fmt= '%.1f';
  unit= 'Hz';
else
  if isempty(ival),
    iv= 1:size(epo.x,1);
  else
    iv= getIvalIndices(ival, epo);
  end
%  fmt= '%d';
  unit= 'ms';
end

if opt.unifiedCLim,
  opt.scalePos= 'none';
end

mnt= mnt_adaptMontage(mnt, epo);

nClasses= size(epo.y, 1);
pat= zeros(nClasses, size(epo.x,2));
for ip= 1:nClasses,
  clInd= find(epo.y(ip,:));  
  pat(ip,:)= mean(mean(epo.x(iv, :, clInd), 1), 3);
end

if length(opt.contour)==1 & opt.contour~=0
  %% choose common levels for contour lines
  dispChans= find(~isnan(mnt.x));
  mi= min(min(pat(:,dispChans)));
  ma= max(max(pat(:,dispChans)));
  opt.contour= goodContourValues(mi, ma, opt.contour);
end

nClasses= size(epo.y, 1);

%%??who made this hack?? Please use input argument 'opt' for such things!
if isfield(epo,'arrangement') & ~isempty(epo.arrangement)
  arr = epo.arrangement;
else
  arr = [1,nClasses];
end

if isempty(opt.ax),
  for ip= 1:nClasses,
    opt.ax(ip)= subplot(arr(1), arr(2), ip);
  end
end

for ip= 1:nClasses,
  axes(opt.ax(ip));
  if length(opt.contour)==1 & opt.contour==0
    [H]= plotScalpPattern(mnt, pat(ip,:), opt);
  else
    [H,lvl]= plotScalpPattern(mnt, pat(ip,:), opt);
  end
  
  h(ip)= H.ax;

  if isfield(epo, 'className'),
    tit= [epo.className{ip} '  '];
  else
    tit= [];
  end
  if isfield(epo, 't'),
%    tit= sprintf(['%s[' fmt ' ' fmt '] ' unit], tit, epo.t(iv([1 end])));
    tit= sprintf(['%s[%g %g] ' unit], tit, trunc(epo.t(iv([1 end])),1));
  end
  if ~isempty(opt.titleAppendix),
    tit= [tit ' ' opt.titleAppendix];
  end
  title(tit);
end

if opt.unifiedCLim,
  unifyCLim(h);
  pos= get(h(end), 'position');
%  shiftAxesLeft(0.08);
%  h_cb= axes('position', [pos(1)+pos(3)+0.01 pos(2) 0.03 pos(4)]);
%  colorbar(h_cb, 'peer', h(end));
  h_cb= colorbar_aside;
  if length(opt.contour)~=1 | opt.contour~=0;
    set(h_cb, 'yTick',lvl);
  end
end

if opt.show_title & isfield(epo, 'title'),
  addTitle(untex(epo.title), 1, 0);
end
