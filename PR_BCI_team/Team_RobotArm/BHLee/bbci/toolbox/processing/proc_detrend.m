function epo= proc_detrend(epo, varargin)

epo.x(:,:)= detrend(epo.x(:,:), varargin{:});
