function plotCSPatterns(fv, mnt, W, la, varargin)
% plotCSPatterns(fv, mnt, W, la, varargin)
% fv is used to derive title for plot,
% mnt contains montage for visualization of CSP patterns
% W contains the patterns 
% la contains labels or other 1d information to label plot

opt= propertylist2struct(varargin{:});
opt= set_defaults(opt, 'colAx','sym');

if isnumeric(fv),
  nClasses= fv;
else
  nClasses= size(fv.y,1);
end
nComps= size(W,2)/nClasses;

clf;
k= 0;
for rr= 1:nClasses,
  for cc= 1:nComps,
    k= k+1;
    subplot(nComps, nClasses, rr+(cc-1)*nClasses);
    scalpPlot(mnt, W(:,k), opt);
    if isempty(la),
      hy= ylabel(sprintf('csp%d', k));
    else
      hy= ylabel(sprintf('csp%d  [%.2f]', k, la(k)));
    end
    set(hy, 'visible','on');
    if cc==1 & isfield(fv, 'className'),
      ht= title(fv.className{rr});
      set(ht, 'fontSize',16, 'fontWeight','bold');
    end
  end
end

if isfield(fv, 'title'),
  addTitle(untex(fv.title), 1, 0);
end
