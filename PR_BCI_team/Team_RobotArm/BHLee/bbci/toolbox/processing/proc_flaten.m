function dat= proc_flaten(dat, varargin)
%dat= proc_flaten(dat)
%
% reshape data matrix to data vector (clash all but last dimensions)
% if an optional parameter force_flaten is given, a single subtrial with 
% size (NxM) will be flatened to NMx1. Default = False.
% use: dat = proc_flaten(dat, 'force_flaten', True);
% 
% added support for single trial flatening (Martijn)
%
% bb, ida.first.fhg.de

opt= propertylist2struct(varargin{:});
opt= set_defaults(opt, ...
  'force_flaten', 0);


if isnumeric(dat),
  % Also flaten the data if it is a plain data matrix
  sz = size(dat);
  dat = reshape(dat, [prod(sz(1:end-1)) sz(end)]); 
elseif isstruct(dat),
  % Old code from the BCI toolbox:
  if isstruct(dat.x),
    dat= proc_flatenGuido(dat);
  else
    sz = size(dat.x);
    if numel(sz) == 2 && opt.force_flaten,
      dat.x = reshape(dat.x, prod(sz), 1);
    else
      dat.x = reshape(dat.x, [prod(sz(1:end-1)) sz(end)]);
    end
  end
else
  error('Don''t know how to flatten this type of data');
end
