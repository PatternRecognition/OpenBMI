function [averagedSMT, rvaluedSMT] = untitled_function(SMT, varargin)
%% Description
% option에 따라 SMT 만들어주는 function

%% main
opt = [varargin{:}];
if ~isstruct(opt) && iscell(opt)
    opt = opt_cellToStruct(opt);
end

if ~isfield(opt, 'MIPlot') opt.MIPlot = 'off'; end
if ~isfield(opt, 'rValue') opt.rValue = false; end
if ~isfield(opt, 'baseline') opt.baseline = []; end
if ~isfield(opt, 'Class') opt.Class = SMT.class{1,2}; end

if isequal(opt.MIPlot, 'on')
    SMT = prep_envelope(SMT);
end
if ~isempty(opt.baseline)
    SMT = prep_baseline(SMT, {'Time', opt.baseline});
end

SMT = prep_selectClass(SMT, {'class', opt.Class});

averagedSMT = prep_average(SMT);

if nargout == 2
    if isequal(opt.rValue, 'on')
        rvaluedSMT = proc_signedrSquare(SMT);
    else
        rvaluedSMT = [];
    end
end