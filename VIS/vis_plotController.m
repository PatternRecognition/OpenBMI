function output = vis_plotController(averaged_SMT, rval_SMT, varargin)
% Hong Kyung, Kim
% hk_kim@korea.ac.kr

%% output_string
output_str = {'';'';'Finished'};

%% Options
opt = opt_cellToStruct(varargin{:});

if ~isfield(opt, 'Interval')
    opt.Interval = [averaged_SMT.ival(1) averaged_SMT.ival(end)];
end
if ~isfield(opt, 'Channels')
    opt.Channels = {averaged_SMT.chan{1}};
end
if ~isfield(opt, 'Class')
    opt.Class = {averaged_SMT.class{1,2}};
end
if ~isfield(opt, 'baseline')
    opt.baseline = [];
end
if ~isfield(opt, 'SelectTime')
    opt.SelectTime = [averaged_SMT.ival(1) averaged_SMT.ival(end)];
end
if ~isfield(opt, 'Align')
    opt.Align = 'vert';
end
if ~isfield(opt, 'TimePlot')
    opt.TimePlot = 'off';
end
if ~isfield(opt, 'TopoPlot')
    opt.TopoPlot = 'off';
end
if ~isfield(opt, 'FFTPlot')
    opt.FFTPlot = 'off';
end
if ~isfield(opt, 'rValue') || isempty(rval_SMT) || isequal(opt.FFTPlot, 'on') % FFT 임시방편
    opt.rValue = 'off';
end

if ~sum(strcmpi({opt.TimePlot, opt.TopoPlot, opt.FFTPlot}, 'on'))
    return;
end

interval = opt.Interval;
chan = opt.Channels;
class = opt.Class;
%% Figure Settings
plts = vis_subplotTemplate(opt);
plt_idx = 1;
%% Pre-processing
% [averaged_SMT, averaged_SMT_r] = untitled_function(SMT, opt);
%% time-domain plot
if isequal(opt.TimePlot, 'on')
    opt.Plots = plts(plt_idx:plt_idx+length(chan) - 1);
    time_plt = vis_timeAveragePlot(averaged_SMT, opt);
    vis_grpYlabel(time_plt, 'Time-Domain');
    plt_idx = plt_idx + length(time_plt);
end

%% FFT Plot
if isequal(opt.FFTPlot , 'on')
    opt.Plots = plts(plt_idx:plt_idx+size(class,1)*length(chan) - 1);
    fft_plt = vis_freqFFTPlot(averaged_SMT, opt);
    vis_grpYlabel(fft_plt, 'FFT');
    plt_idx = plt_idx + size(class,1)*length(chan);
end
%% ERSP plot
if isfield(opt, 'ErspPlot') && isequal(opt.ErspPlot, 'on')
    %%TODO: Developing the ERSP graph
end
%% Topo plot
if isequal(opt.TopoPlot, 'on')
    opt.Plots = plts(plt_idx:plt_idx+size(class,1)*size(interval, 1) - 1);
    topo_plt = vis_topoPlot(averaged_SMT, opt);
    vis_grpYlabel(topo_plt, 'Topography');
    plt_idx = plt_idx + length(topo_plt);
end
%% R-value
if isequal(opt.rValue, 'on')
    opt.Plots = plts(plt_idx:plt_idx+(isequal(opt.TopoPlot, 'on')*size(interval, 1))-1+isequal(opt.TimePlot, 'on'));
    r_plt = vis_rValuePlot(rval_SMT, opt);
    vis_grpYlabel(r_plt, 'r-value');
    plt_idx = plt_idx + length(r_plt);
end
output = output_str;
end