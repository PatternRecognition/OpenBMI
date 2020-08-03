function varargout = proc_downsample(dat, varargin)
% proc_downsample - Downsample by subsampling
%
% Synopsis:
% cnt_d          = proc_downsample(dat, n)
% [cnt_d, mrk_d] = proc_downsample(dat, mrk, n)
%
% Ryota Tomioka

if isstruct(varargin{1})
  mrk = varargin{1};
  n = varargin{2};
  
  if mrk.fs~=dat.fs
    error('Sampling frequency mismatching.');
  end
  
else
  n = varargin{1}
end


dat.x = dat.x(1:n:end,:);
dat.fs = dat.fs/n;
varargout{1} = dat;

if exist('mrk','var')
  mrk.pos = ceil(mrk.pos/n);
  mrk.fs = dat.fs;
  varargout{2} = mrk;
end
  