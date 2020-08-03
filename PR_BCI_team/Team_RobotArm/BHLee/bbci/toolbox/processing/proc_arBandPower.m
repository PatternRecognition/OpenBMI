function bp= proc_arBandPower(fv, band, varargin)

if length(varargin)==1,
  opt= struct('order', varargin{1});
else
  opt= propertylist2struct(varargin{:});
end
opt= set_defaults(opt, ...
                  'order', [], ...
                  'armethod', 'yule', ...
                  'bins', 4*diff(band)+1);

w= linspace(band(1), band(2), opt.bins) / fv.fs * 2*pi;
E= exp( -i * ([0:opt.order]' * w ));
[T, nChans, nEvents]= size(fv.x);

bp= copy_struct(fv, 'not','x');
bp.x= zeros([1 nChans nEvents]);
for ce= 1:nChans*nEvents,
  [ar,noisevar]= feval(['ar' opt.armethod], fv.x(:,ce), opt.order);
  bp.x(1, ce)= mean( noisevar ./ (abs( ar * E ).^2) );
end
