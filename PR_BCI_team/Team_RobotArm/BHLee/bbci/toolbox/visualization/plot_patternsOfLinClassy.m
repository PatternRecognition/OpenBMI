function hnd= plot_patternsOfLinClassy(mnt, C, fv, varargin)
%hnd= plot_patternsOfLinClassy(mnt, C, fv, <opt>)
%
% IN   mnt - electrode montage
%      C   - trained classifier as given by
%            train_{LSR, LDA, FisherDiscriminant, MSM1}
%      fv  - feature vectors

if ~exist('opt','var'), 
  opt= struct('scalePos','vert'); 
else
  opt= propertylist2struct(varargin{:});
end
if isfield(fv, 'xUnit'),
  default_xUnit= fv.xUnit;
else
  default_xUnit= 'ms';
end

opt= set_defaults(opt, ...
                  'resolution', 10, ...
                  'colAx', [min(C.w) max(C.w)], ...
                  'xUnit', default_xUnit);

nChans= length(fv.clab);
nPatterns= length(C.w)/nChans;
wpat= mnt;
wpat.x(chanind(mnt, 'not', fv.clab{:}))= NaN;
clf;
for ip= 1:nPatterns,
  hnd.ax(ip)= suplot(nPatterns, ip, 0.08, [0 0.05 0.05]);
  plotScalpPattern(wpat, C.w(ip+(0:nChans-1)*nPatterns), opt);
  if isfield(fv, 't'),
    hnd.subtitle(ip)= title(sprintf('[%d %s]', fv.t(ip), opt.xUnit));
  end
end
if isfield(fv, 'title'),
  hnd.title= addTitle(untex(fv.title), 1);
end
