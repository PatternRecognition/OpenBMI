function spec= proc_arspectrum(fv, band, varargin)

if length(varargin)==1,
  opt= struct('order', varargin{1});
else
  opt= propertylist2struct(varargin{:});
end
opt= set_defaults(opt, ...
                  'order', [], ...
                  'armethod', 'yule', ...
                  'bins', 4*diff(band)+1, ...
                  'scaling','db');

spec= copy_struct(fv, 'not','x');
spec.t= linspace(band(1), band(2), opt.bins);
w= spec.t / fv.fs * 2*pi;
E= exp( -i * ([0:opt.order]' * w ));
[T, nChans, nEvents]= size(fv.x);

spec.x= zeros([opt.bins nChans nEvents]);
for ce= 1:nChans*nEvents,
  [ar,noisevar]= feval(['ar' opt.armethod], fv.x(:,ce), opt.order);
%  spec.x(:, ce)= noisevar ./ (abs( ar * E ).^2);
  spec.x(:, ce)= 1 ./ (abs( ar * E ).^2);
end

switch(lower(opt.scaling)),
 case 'db',
  spec.x= 10 * log10(spec.x);
  spec.yUnit= 'dB';
 case 'power',
  spec.yUnit= 'power';
 otherwise,
  error('unknown scaling');
end

spec.xUnit= 'Hz';
