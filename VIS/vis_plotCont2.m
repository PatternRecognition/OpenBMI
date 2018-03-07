function output = vis_plotCont2(SMT, varargin)
% Description:
%   Draw  scalp topographies for all selected intervals,separately for each each class.
%   Scalp topographies of each classe are plotted in one row, and shared the same color map
%   scaling in each classes.
%
% Example Code:
%    visual_scalpPlot(SMT,CNT, {'Ival' , [start interval : time increase parameter: end intercal]});
%
% Input:
%   visual_scalpPlot(SMT,CNT, <OPT>);
%   SMT: Data structrue (ex) Epoched data structure
%   CNT: Continuous data structure
%
% Option:
%      .Ival - Selecting the interested time interval depending on time increase parameter
%                 (e.g. {'Ival' [ [-2000 : 1000: 4000]])
%
% Return:
%   Scalp topographies
%
% See also:
%    opt_getMontage, opt_cellToStruct
%
% Ji Hoon, Jeong
% jh_jeong@korea.ac.kr
% Hong Kyung, Kim
% hk_kim@korea.ac.kr

output_str = {'';'';'Finished'};

%%Options
opt = opt_cellToStruct(varargin{:});

if isfield(opt, 'Interval') 
    interval = opt.Interval; 
else    
    opt.Interval = [SMT.ival(1) SMT.ival(end)];
    interval = opt.Interval; 
end
if isfield(opt, 'Range') 
    plot_range = opt.Range; 
else
    opt.Range = 'sym'; 
    plot_range = opt.Range;
end
if isfield(opt, 'Channels') 
    chan = opt.Channels; 
else
    opt.Channels = {SMT.chan{1:2}}; 
    chan = opt.Channels;
end
if isfield(opt, 'Class') 
    class = opt.Class; 
else
    opt.Class = {SMT.class{1:2,2}}; 
    class = opt.Class; 
end
if isfield(opt, 'Patch') 
    Patch = opt.Patch; 
else
    opt.Patch = 'on';
    Patch = opt.Patch; 
end
if isfield(opt, 'TimePlot') 
    TimePlot = opt.TimePlot; 
else
    opt.TimePlot = 'off';
    TimePlot = opt.TimePlot; 
end
if isfield(opt, 'ErspPlot') 
    ErspPlot = opt.ErspPlot;
else
    opt.ErspPlot = 'off';
    ErspPlot = opt.ErspPlot;
end
if isfield(opt, 'ErdPlot') 
    ErdPlot = opt.ErdPlot; 
else
    opt.ErdPlot = 'off'; 
    ErdPlot = opt.ErdPlot; 
end
if isfield(opt, 'TopoPlot') 
    TopoPlot = opt.TopoPlot; 
else
    opt.TopoPlot = 'on'; 
    TopoPlot = opt.TopoPlot; 
end
if isfield(opt, 'RPlot') 
    RPlot = opt.RPlot; 
else
    opt.RPlot = 'on'; 
    RPlot = opt.RPlot; 
end
if isfield(opt, 'FFTPlot') 
    FFTPlot = opt.FFTPlot; 
else
    opt.FFTPlot = 'off'; 
    FFTPlot = opt.FFTPlot; 
end
if isfield(opt, 'Baseline') 
    baseline = opt.Baseline; 
else
     opt.Baseline = [0 0]; 
    baseline = opt.Baseline; 
end
if ~isfield(opt, 'SelectTime')
    opt.SelectTime = [SMT.ival(1) SMT.ival(end)]; 
end
if isfield(opt, 'Align') 
    Align = opt.Align; 
else
    opt.Align = 'vert'; 
    Align = opt.Align; 
end
if ~isfield(opt, 'Envelope')
    opt.Envelope = false;
end
%% Figure Settings
plts = vis_subplotTemplate(opt);
plt_idx = 1;
%% time-domain plot
if isequal(TimePlot, 'on')
    time_plt = vis_timeAveragePlot(plts(plt_idx:plt_idx+length(chan) - 1),SMT, opt);
    vis_grpYlabel(time_plt, 'Time-Domain');
    plt_idx = plt_idx + length(chan);
end
%% ERDERS plot
if isequal(ErdPlot, 'on')
    opt.Envelope = true;
    erders_plt = vis_timeAveragePlot(plts(plt_idx:plt_idx+length(chan) - 1),SMT, opt);
    vis_grpYlabel(erders_plt, 'ERD/ERS');
    plt_idx = plt_idx + length(chan);
end
if isequal(RPlot, 'on')
    r_plt = vis_rValuePlot(plts(plt_idx:plt_idx+length(chan) - 1), SMT, opt);
    vis_grpYlabel(r_plt, 'r-Value');
    plt_idx = plt_idx + length(chan);   
end
%% FFT Plot
if isequal(FFTPlot , 'on')
    fft_plt = vis_freqFFTPlot(plts(plt_idx:plt_idx+size(class,1)*length(chan) - 1), SMT, opt);
    vis_grpYlabel(fft_plt, 'FFT');
    plt_idx = plt_idx + size(class,1)*length(chan);
end
%% ERSP plot
if isequal(ErspPlot, 'on')
    %%TODO: Developing the ERSP graph
end
%% Topo plot
if isequal(TopoPlot, 'on')
    topo_plt = vis_topoPlot(plts(plt_idx:plt_idx+size(class,1)*size(interval, 1) - 1),SMT, opt);
    vis_grpYlabel(topo_plt, 'Topography');
end
output = output_str;
end