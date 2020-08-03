function yLim= selectYLim(h, varargin)

opt= propertylist2struct(varargin{:});
opt= set_defaults(opt, ...
                  'policy', 'auto', ...
                  'tightenBorder', 0.03, ...
                  'symmetrize', 0, ...
                  'setLim', 1);

switch(lower(opt.policy)),
 case 'auto',
  yLim= get(h, 'YLim');
 case 'tightest',
  backaxes(h);
  axis('tight');
  yLim= get(h, 'yLim');
 case 'tight',
  backaxes(h);
  axis('tight');
  yl= get(h, 'yLim');
  %% add border not to make it too tight:
  yl= yl + [-1 1]*opt.tightenBorder*diff(yl);
  %% determine nicer limits
  dig= floor(log10(diff(yl)));
  if diff(yl)>1,
    dig= max(1, dig);
  end
  yLim= [trunc(yl(1),-dig+1,'floor') trunc(yl(2),-dig+1,'ceil')];
 otherwise,
  error('unknown policy');
end

if opt.symmetrize,
  ma= max(abs(yLim));
  yLim= [-ma ma];
end

if opt.setLim,
  set(h, 'YLim',yLim);
end

if nargout==0,
  clear yLim;
end
