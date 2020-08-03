function [ H1 H2 H3 H4 ] = stdERPplots(epo, epo_r, varargin)
%STDERPPLOTS Summary of this function goes here
%   Detailed explanation goes here

% Johannes Hoehne 10.2011 j.hoehne@tu-berlin.de

opt= propertylist2struct(varargin{:});
opt= set_defaults(opt, ...
    'ivals',  [], ... %if not specified, ivals are taken by the heuristic
    'clab', {'CPz'}, ...
    'plot_dir', [], ...
    'plot_prefix', 'stdERPanalysis_', ...
    'plot_format', 'eps', ...
    'colOrder', [1 0 1; 0.4 0.4 0.4], ...
    'opt_scalp', {'resolution', 30, 'extrapolate', 1}, ...
    'visualize_score_matrix', 1, ...
    'grid_plot', 1, ...
    'scalpEvolutionPlusChannel', 1, ...
    'scalpEvolutionPlusChannelPlusRsquared', 1, ...
    'collapseToOneFig', 1, ...
    'grd', 'medium' ...
    );

if ~isempty(opt.plot_dir)
    opt.opt_fig = strukt('folder', opt.plot_dir, 'prefix', opt.plot_prefix, 'format', opt.plot_format);
end

mnt= getElectrodePositions(epo.clab);
mnt= mnt_setGrid(mnt, opt.grd);

% grd= sprintf(['EOGh,scale,AF3,,AF4,legend,EOGv\n' ...
%     ',F3,F1,Fz,F2,F4,,\n' ...
%     'C5,C3,C1,Cz,C2,C4,C6\n' ...
%     'P5,P3,P1,Pz,P2,P4,P6\n' ...
%     'P9,PO7,O1,Oz,O2,PO8,P10']);
%      mnt= mnt_setGrid(mnt, grd);
mnt= mnt_excenterNonEEGchans(mnt, 'E*');

%% determine ivals
if isempty(opt.ivals)
    epo_r_smooth = proc_movingAverage(epo_r, 50, 'method', 'centered'); %smoothing to prevent artifactual local minima
    [opt.ivals, nfo]= ...
        select_time_intervals(epo_r_smooth, 'visualize', 0, 'visu_scalps', 0, ...
        'clab',{'not','E*','Fp*','AF*', 'Ref'});
    opt.ivals= visutil_correctIvalsForDisplay(opt.ivals, 'fs',epo.fs);
    clear epo_r_smooth;
end


%% visualize_score_matrix
if opt.visualize_score_matrix
    hf1=figure();
    H1 = visualize_score_matrix(epo_r, opt.ivals);
    if ~isempty(opt.plot_dir)
        printFigure('r_matrix', [15 12], opt.opt_fig);
    end
    else, hf1 = [];
end


%% grid_plot
if opt.grid_plot
    hf2=figure();
    H2= grid_plot(epo, mnt, defopt_erps, 'colorOrder',opt.colOrder);
    pause(1)
    try
        grid_addBars(epo_r, 'h_scale',H2.scale);
    catch
        H2= grid_plot(epo, mnt, defopt_erps);
        grid_addBars(epo_r, 'h_scale',H2.scale);
    end
    if ~isempty(opt.plot_dir)
        printFigure(['erp'], [19 12], opt.opt_fig);
    end
    else, hf2 = [];
end

%% scalpEvolutionPlusChannel
if opt.scalpEvolutionPlusChannel
    hf3=figure();
    H3 = scalpEvolutionPlusChannel(epo_r, mnt, opt.clab, opt.ivals, ...
        defopt_scalp_r, opt.opt_scalp{:}, ...
        'lineWidth', 2, ...
        'channelAtBottom',1, ...
        'legend_pos',0);
    if ~isempty(opt.plot_dir)
        printFigure(['erp_topo_r'], [20 4+5], opt.opt_fig);
    end
else, hf3 = [];
end

%% scalpEvolutionPlusChannelPlusRsquared
if opt.scalpEvolutionPlusChannelPlusRsquared
    
    dum = scalpEvolutionPlusChannelPlusRsquared(epo, epo_r, mnt, opt.clab, opt.ivals, ...
        defopt_scalp_erp, opt.opt_scalp{:}, ...
        'legend_pos',2,  'colorOrder',opt.colOrder);
    hf4 = gcf;
    grid_addBars(epo_r);
    if ~isempty(opt.plot_dir)
        printFigure(['erp_topo'], [20 15], opt.opt_fig);
    end
    else, hf4 = [];
end
varargout = {hf1, hf2, hf3, hf4};
if or (opt.collapseToOneFig, nargout == 1)
    varargout = fig2subplot([hf1 hf2 hf3 hf4],'rowscols',[2 2],'label','(a)');
    close([hf1 hf2 hf3 hf4]);
    if ~isempty(opt.plot_dir)
        printFigure(['Overview_ERP'], [25 25], opt.opt_fig);
    end
end

