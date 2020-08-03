function fv= proc_catFeatures(fv, varargin)
%fv= proc_catFeatures(fv, fv2, <fv3, ...>)
%
% concatenate feature vectors

% bb 03/03, ida.first.fhg.de

if length(varargin)>1,
  for ii= 1:length(varargin),
    fv= proc_catFeatures(fv, varargin{ii});
  end
  return
else
  if isempty(fv),
    fv= varargin{1};
    return;
  end
  fv2= varargin{1};
end

sz= size(fv.x);
s2= size(fv2.x);
if ~isequal(sz(2:end), s2(2:end)),
  msg= 'inconsistent feature dimensionality, concat flatened features';
  bbci_warning(msg, 'processing', mfilename);
  fv= proc_flaten(fv);
  fv2= proc_flaten(fv2);
end

%% is this the only important action
fv.x= cat(1, fv.x, fv2.x);

if isfield(fv,'t')
  if isfield(fv2,'t'),
    fv.t= cat(1, fv.t(:), fv2.t(:));
  else
    fv= rmfield(fv, 't');
  end
end
