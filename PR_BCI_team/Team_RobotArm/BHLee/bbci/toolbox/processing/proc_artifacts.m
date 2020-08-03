function dat = proc_artifacts(dat, varargin)
% see find_artifacts for documentation

% by guido


ev = find_artifacts(dat, varargin{:});
dat.y(:,ev) = 0;
