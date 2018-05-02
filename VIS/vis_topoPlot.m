function grp_plots = vis_topoPlot(SMT, varargin)
% Description:
%
%
%

%% Options
opt = [varargin{:}];
if ~isstruct(opt) && iscell(opt)
    opt = opt_cellToStruct(opt);
end

output_str = [];
if ~isfield(opt, 'Colormap') opt.Colormap = 'parula'; end
if ~isfield(opt, 'Quality') opt.Quality = 'high'; end
if ~isfield(opt, 'Class') opt.Class = {SMT.class{1,2}}; end
if ~isfield(opt, 'Baseline') opt.Baseline = [SMT.ival(1) SMT.ival(1)]; end
if ~isfield(opt, 'SelectTime') opt.SelectTime = [SMT.ival(1) SMT.ival(end)]; end
if ~isfield(opt, 'Interval') opt.Interval = [opt.SelectTime(1) opt.SelectTime(end)]; end
if ~isfield(opt, 'Range') opt.Range = 'sym'; end
if ~isfield(opt, 'Plots') opt.Plots = gca; end

grp_plots = opt.Plots;

% scalp_plot quality
switch lower(opt.Quality)
    case 'high'
        resol = 256;
    case 'medium'
        resol = 144;
    case 'low'
        resol = 32;
    otherwise
        resol = 256;
end
% Scalp colormap;
colormap(opt.Colormap);

%% Creating Montage
MNT = opt_getMontage(SMT);

if ~isequal(MNT.chan, SMT.chan)
    if length(SMT.chan) > length(MNT.chan)
        tmp = SMT.chan(~ismember(SMT.chan, MNT.chan));
        output_str = tmp{1};
        for i = 2:length(tmp)
            output_str = [output_str sprintf(', %s',tmp{i})];
        end
        output_str = {''; [output_str, ' are missed']};
    end
    [a, tmp] = ismember(MNT.chan, SMT.chan);
    SMT.x = SMT.x(:,:,tmp);
    SMT.chan = SMT.chan(tmp);
    clear tmp;
end

idx = 1;
for i = 1: size(opt.Class, 1)
    ivalSegment = size(opt.Interval,1);
    topo_range = zeros(ivalSegment,2);
    
    for seg = 1: ivalSegment
        SMTintervalstart = find(SMT.ival == opt.Interval(seg,1));
        SMTintervalEnd = find(SMT.ival == opt.Interval(seg,2));
        ivalSMT = squeeze(SMT.x(SMTintervalstart:SMTintervalEnd,i,:));
        w{i, seg} = mean(ivalSMT,1);
        topo_range(seg, :) = minmax(w{i,seg});
    end
    %% range_options
    if ~isfloat(opt.Range)
        switch opt.Range
            case 'sym'
                p_range = [-abs(max(topo_range(:,2))), abs(max(topo_range(:,2)))];
            case '0tomax'
                p_range = [0.0001*diff([min(topo_range(:,1)) max(topo_range(:,2))]), max(topo_range(:,2))];
            case 'minto0'
                p_range = [min(topo_range(:,1)), 0.0001*diff([min(topo_range(:,1)) max(topo_range(:,2))])];
            case 'mintomax'
                p_range = [min(topo_range(:,1)) max(topo_range(:,2))];
            case 'mean'
                p_range = [mean(topo_range(:,1)) mean(topo_range(:,2))];
        end
    else
        p_range = opt.Range;
    end
    if diff(p_range)==0
        p_range(2)= p_range(2)+eps;
    end
    %% Draw
    for seg = 1: ivalSegment
        plot_scalp(grp_plots(idx), w{i, seg}, MNT, p_range, resol);
        xlabel(grp_plots(idx), sprintf('[%d - %d] ms',opt.Interval(seg,:)), 'FontWeight', 'normal');
        idx = idx + 1;
    end
    % Keeping scalp size;
    last_position = get(grp_plots(idx - 1), 'Position');
    colorbar(grp_plots(idx - 1), 'vert');
    tmp = get(grp_plots(idx - 1), 'Position');
    tmp(3:4) = last_position(3:4);
    set(grp_plots(idx - 1),'Position',tmp);
    set(get(grp_plots(idx - size(opt.Interval, 1)),'Ylabel'), 'Visible','on', 'String', {opt.Class{i};''}, ...
        'FontWeight', 'normal', 'FontSize', 12);
end
end