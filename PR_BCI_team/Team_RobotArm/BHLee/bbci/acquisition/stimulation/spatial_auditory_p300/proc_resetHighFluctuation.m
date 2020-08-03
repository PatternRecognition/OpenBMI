function data = proc_resetHighFluctuation(data, varargin),

  opt= propertylist2struct(varargin{:});
  opt= set_defaults(opt, ...
      'threshold', 70);

  highVal = max(max(abs(data.x)));
  if highVal > opt.threshold,
    data.x = data.x * NaN;
  end
    
    
end