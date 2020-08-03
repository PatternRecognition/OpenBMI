function adapt_trigger_stim(opt, varargin),

  opt= propertylist2struct(opt, varargin{:}{:});
%   opt2= set_defaults(opt, ...
%     'test', [], ...
%     'nrExtraStimuli', 0);

  if isfield(opt, 'tact_duration') && opt.tact_duration,
    ppTrigger(opt.tact_trig_offset + opt.clsOut);
    send_trigger_vest('send', opt.tact_map(opt.clsOut));
  end 
end