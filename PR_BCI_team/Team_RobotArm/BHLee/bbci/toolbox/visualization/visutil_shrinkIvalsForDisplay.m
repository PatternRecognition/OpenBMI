function ival= visutil_shrinkIvalsForDisplay(ival, dat, varargin)

opt= propertylist2struct(varargin{:});
opt= set_defaults(opt, ...
                  'border', 0.25, ...
                  'border_unit', 'absolute');

switch(opt.border_unit),
 case {'absolute','abs'},
  dd= opt.border * median(diff(dat.t));
 case {'relative','rel'},
  dd= opt.border;
 otherwise,
  error('unknown value for border_unit');
end

to_be_checked= 1:size(ival, 1);
for ii= to_be_checked,
  iv= getIvalIndices(ival(ii,:), dat, 'ival_policy','minimal');
  ival(ii,:)= dat.t(iv([1 end])) + [-1 1]*dd;
end
