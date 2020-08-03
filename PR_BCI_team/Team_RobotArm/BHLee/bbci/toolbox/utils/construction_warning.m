function construction_warning(name, varargin)

persistent last_warning

if length(varargin)==1,
  opt= struct('interval', varargin{1});
else
  opt= propertylist2struct(varargin{:});
  opt= set_defaults(opt, ...
                    'interval', 10*60);
end

this_warning= clock;
if ~isempty(last_warning),
  if etime(this_warning, last_warning) < opt.interval,
    return;
  end
end

last_warning= this_warning;
warning('File %s is in construction and may change in future.', name);
