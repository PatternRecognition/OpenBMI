function [out_str, out_plts] = vis_plotController2(averaged_SMT, rval_SMT, varargin)
% Hong Kyung, Kim
% hk_kim@korea.ac.kr
%% output_string
out_str = {'';'';'Finished'};

if nargin == 2
    varargin = {rval_SMT};
    rval_SMT = [];
end

%% Options
opt = opt_cellToStruct(varargin{:});
def_opt = struct('interval', [averaged_SMT.ival(1) averaged_SMT.ival(end)],...
    'channels', averaged_SMT.chan(1), 'class', {averaged_SMT.class{1,2}}, ...
    'baseline', [], 'selecttime', [averaged_SMT.ival(1) averaged_SMT.ival(end)],...
    'align', 'vert', 'timeplot', 'off', 'topoplot', 'off', 'fftplot', 'off', 'rvalue', 'off');
opt = opt_defaultParsing(def_opt, opt);

if isempty(rval_SMT) || isequal(opt.fftplot, 'on') % FFT 임시방편
    opt.rvalue = 'off';
end

if ~any(strcmpi({opt.timeplot, opt.topoplot, opt.fftplot}, 'on'))
    return
end

interval = opt.interval;
chan = opt.channels;
class = opt.class;
%% Figure Settings
plts = vis_subplotTemplate(opt);
plt_idx = 1;
%% Pre-processing
% [averaged_SMT, averaged_SMT_r] = untitled_function(SMT, opt);
%% time-domain plot
if isequal(opt.timeplot, 'on')
    opt.plots = plts(plt_idx:plt_idx+length(chan) - 1);
    time_plt = vis_timeAveragePlot(averaged_SMT, opt);
    vis_grpYlabel(time_plt, 'Time-Domain');
    plt_idx = plt_idx + length(time_plt);
end
%% FFT Plot
if isequal(opt.fftplot , 'on')
    opt.plots = plts(plt_idx:plt_idx+size(class,1)*length(chan) - 1);
    fft_plt = vis_freqFFTPlot(averaged_SMT, opt);
    vis_grpYlabel(fft_plt, 'FFT');
    plt_idx = plt_idx + size(class,1)*length(chan);
end
%% ERSP plot
if isfield(opt, 'erspplot') && isequal(opt.ErspPlot, 'on')
    %%TODO: Developing the ERSP graph
end
%% Topo plot
if isequal(opt.topoplot, 'on')
    opt.plots = plts(plt_idx:plt_idx+size(class,1)*size(interval, 1) - 1);
    topo_plt = vis_topoPlot(averaged_SMT, opt);
    vis_grpYlabel(topo_plt, 'Topography');
    plt_idx = plt_idx + length(topo_plt);
end
%% R-value
if isequal(opt.rvalue, 'on')
    opt.plots = plts(plt_idx:plt_idx+(isequal(opt.topoplot, 'on')*size(interval, 1))-1+isequal(opt.timeplot, 'on'));
    r_plt = vis_rValuePlot(rval_SMT, opt);
    vis_grpYlabel(r_plt, 'r-value');
    plt_idx = plt_idx + length(r_plt);
end
out_plts = plts;
end