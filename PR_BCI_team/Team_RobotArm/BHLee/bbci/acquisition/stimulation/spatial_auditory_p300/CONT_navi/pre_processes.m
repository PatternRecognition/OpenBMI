function output = post_processes(currentState, Lut, Dict, history, varargin),
%PRE_PROCESSES Summary of this function goes here
%   Detailed explanation goes here

opt= propertylist2struct(varargin{:});
opt= set_defaults(opt, ...
    'test', []);

labels = Lut(currentState).direction;
labels = rmfield(labels, 'nState');

output = opt.procVar;