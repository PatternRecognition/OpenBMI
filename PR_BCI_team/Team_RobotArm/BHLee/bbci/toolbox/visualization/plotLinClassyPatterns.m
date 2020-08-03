function hp= plotLinClassyPatterns(mnt, C, fv, varargin)
%hp= plotLinClassyPatterns(mnt, C, fv, opts)
%
% IN   mnt  - electrode montage
%      C    - trained classifier as given by
%             train_{LSR, LDA, FisherDiscriminant, MSM1}
%      fv   - feature vectors
%      opts - options to plotScalpPattern
%
% SEE plotScalpPatterns

default_opts= struct('scalePos','horiz', 'colAx','sym');
nChans= length(fv.clab);
nPatterns= length(C.w)/nChans;
wpat= mnt;
wpat.x(chanind(mnt, 'not', fv.clab{:}))= NaN;
clf;
for ip= 1:nPatterns,
  hp(ip)= subplotxl(1, nPatterns, ip, 0.1, 0.01);
  plotScalpPattern(wpat, C.w(ip+(0:nChans-1)*nPatterns), ...
                   default_opts, varargin{:});
  if isfield(fv, 't'),
    ht= title(sprintf('[%d ms]', fv.t(ip)));
  end
end
