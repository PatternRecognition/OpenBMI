function fv= proc_peakArea(fv, ival, varargin);

if length(varargin)==1 & isnumeric(varargin{1}),
  opt= struct('refIval', varargin{1});
else
  opt= propertylist2struct(varargin{:});
end
opt= set_defaults(opt, 'refIval', [], ...
                       'interpolation_ival', [], ...
                       'relative', 0);

if isempty(opt.interpolation_ival) & isempty(opt.refIval),
  opt.refIval= ival;
end

iv= getIvalIndices(ival, fv);
if ~isempty(opt.refIval),
  iv2= getIvalIndices(opt.refIval, fv);
end
if ~isempty(opt.interpolation_ival),
  iv3= getIvalIndices(opt.interpolation_ival, fv);
  if ~isempty(setdiff(iv,iv3)),
    error(sprintf('interpolation_ival [%g %g] must encompass ival [%g %g]', ...
      opt.interpolation_ival, ival));
  end
end

[T, nChans, nEvents]= size(fv.x);
nCE= nChans*nEvents;
area= zeros(1, nCE);
for ii= 1:nCE,
  if ~isempty(opt.refIval),
    ref= linspace(fv.x(iv2(1),ii), fv.x(iv2(end),ii), length(iv2));
    hub= fv.x(iv,ii) - ref';
  end
  if ~isempty(opt.interpolation_ival),
    ref= zeros(size(fv.x,1), 1);
    ref(iv3)= linspace(fv.x(iv3(1),ii), fv.x(iv3(end),ii), length(iv3));
    hub= fv.x(iv,ii) - ref(iv);
  end
  area(ii)= sum(hub);
  [fv.peak(ii), mi]= max(hub);
  fv.peak_time(ii)= fv.t(iv(1)+mi-1);
end
if opt.relative,
  area= area / length(iv);
end

fv.x= reshape(area, [1 nChans nEvents]);
fv= rmfield(fv, {'t', 'xUnit', 'yUnit'});
