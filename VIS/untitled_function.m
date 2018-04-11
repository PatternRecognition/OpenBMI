function [averagedSMT, rvaluedSMT] = untitled_function(SMT, opt)
%% Description
% option에 따라 SMT 만들어주는 function

%% main
if ~isstruct(opt)
    opt = opt_cellTostruct(opt);
end
if ~isfield(opt, 'envelope') opt.envelope = false; end
if ~isfield(opt, 'rValue') opt.rValue = false; end
if ~isfield(opt, 'baseline') opt.baseline = []; end

if opt.envelope
    SMT = prep_envelope(SMT);
end
if ~isempty(opt.baseline)
    SMT = prep_baseline(SMT, {'Time', opt.baseline});
end

averagedSMT = prep_average(SMT);

if nargout == 2
    if isequal(opt.rValue, 'on')
        rvaluedSMT = proc_rSquareSigned(SMT);
    else
        rvaluedSMT = [];
    end
end