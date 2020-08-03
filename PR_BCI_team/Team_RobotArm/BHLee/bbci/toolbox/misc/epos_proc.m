function epos= epos_proc(epos, fcn, varargin)

sz= size(epos);
epos= reshape(epos, [1 numel(epos)]);

for k= 1:length(epos),
  epos{k}= feval(['proc_' fcn], epos{k}, varargin{:});
end

epos= reshape(epos, sz);
