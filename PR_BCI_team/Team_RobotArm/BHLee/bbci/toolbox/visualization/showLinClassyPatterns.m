function H= showLinClassyPatterns(mnt, C, fv, varargin)
%showLinClassyPatterns(mnt, C, fv)
%
% IN   mnt - electrode montage
%      C   - trained classifier as given by
%            train_{LSR, LDA, FisherDiscriminant, MSM1}
%      fv  - feature vectors


opt= propertylist2struct(varargin{:});
[opt, isdefault]= ...                 
    set_defaults(opt, ...
                 'colormap',cmap_posneg(51), ...
                 'titles', []);
if isempty(opt.titles) & isfield(fv, 't'),
  opt.title= cprintf('[%g ms]', fv.t');
end

mm= max(abs(C.w));

nChans= length(fv.clab);
nPatterns= length(C.w)/nChans;
wpat= mnt;
wpat.x(chanind(mnt, 'not', fv.clab{:}))= NaN;
clf;
opt.scalePos= 'none';
for ip= 1:nPatterns,
  H(ip).ax= subplotxl(1, nPatterns, ip, 0.1, [0.01 0.01 0.1]);
  H(ip).scalp= scalpPlot(wpat, C.w(ip+(0:nChans-1)*nPatterns), opt, ...
                         'colAx', [-mm mm]);
  if ip==nPatterns,
    colorbar_aside;
  end
  if ~isempty(opt.titles),
    H(ip).title= title(opt.titles{ip});
  end
end
