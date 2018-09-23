function grp_plots = vis_topoPlot(dat, varargin)
% Description:
%
%
%

%% Options
if iscell(varargin{:})
    opt = opt_cellToStruct(varargin);
elseif isstruct(varargin{:})
    opt = varargin{:};
end

def_opt = struct('colormap', 'parula', 'quality', 'high', 'class', {dat.class{1,2}},...
    'baseline', [dat.ival(1) dat.ival(1)], 'selecttime', [dat.ival(1) dat.ival(end)],...
    'interval', [opt.selecttime(1) opt.selecttime(end)], 'range', 'sym', ...
    'plots', [], 'topoplot', 'on');
opt = opt_defaultParsing(def_opt, opt);

% output_str = [];
% if ~isfield(opt, 'colormap') opt.colormap = 'parula'; end
% if ~isfield(opt, 'quality') opt.quality = 'high'; end
% if ~isfield(opt, 'class') opt.class = {dat.class{1,2}}; end
% if ~isfield(opt, 'baseline') opt.baseline = [dat.ival(1) dat.ival(1)]; end
% if ~isfield(opt, 'selecttime') opt.selecttime = [dat.ival(1) dat.ival(end)]; end
% if ~isfield(opt, 'interval') opt.interval = [opt.selecttime(1) opt.selecttime(end)]; end
% if ~isfield(opt, 'range') || isempty(opt.range) opt.range = 'sym'; end
% if ~isfield(opt, 'plots') opt.plots = gca; end
if isempty(opt.plots)
    grp_plots = vis_subplotTemplate(opt);
else
    grp_plots = opt.plots;
end
% scalp_plot quality
switch lower(opt.quality)
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
colormap(opt.colormap);

%% Creating Montage
MNT = opt_getMontage(dat);

if ~isequal(MNT.chan, dat.chan)
    if length(dat.chan) > length(MNT.chan)
        tmp = dat.chan(~ismember(dat.chan, MNT.chan));
        output_str = tmp{1};
        for i = 2:length(tmp)
            output_str = [output_str sprintf(', %s',tmp{i})];
        end
        output_str = {''; [output_str, ' are missed']};
    end
    [~, tmp] = ismember(MNT.chan, dat.chan);
    dat.x = dat.x(:,:,tmp);
    dat.chan = dat.chan(tmp);
    clear tmp;
end

%% opt.clas 순서대로
[~, selected_class_order] = ismember(opt.class, dat.class(:,2));

idx = 1;
for i = selected_class_order'
    ivalSegment = size(opt.interval,1);
    topo_range = zeros(ivalSegment,2);
    
    for seg = 1: ivalSegment
        SMTintervalstart = find(dat.ival == opt.interval(seg,1));
        SMTintervalEnd = find(dat.ival == opt.interval(seg,2));
        ivalSMT = squeeze(dat.x(SMTintervalstart:SMTintervalEnd,i,:));
        w{i, seg} = mean(ivalSMT,1);
        topo_range(seg, :) = minmax(w{i,seg});
    end
    %% range_options
    if ~isfloat(opt.range)
        switch opt.range
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
        p_range = opt.range;
    end
    if diff(p_range)==0
        p_range(2)= p_range(2)+eps;
    end
    %% Draw
    for seg = 1: ivalSegment
        plot_scalp(grp_plots(idx), w{i, seg}, MNT, p_range, resol);
        xlabel(grp_plots(idx), sprintf('[%d - %d] ms',opt.interval(seg,:)), 'FontWeight', 'normal');
        idx = idx + 1;
    end
    % Keeping scalp size;
    last_position = get(grp_plots(idx - 1), 'Position');
    colorbar(grp_plots(idx - 1), 'vert');
    tmp = get(grp_plots(idx - 1), 'Position');
    tmp(3:4) = last_position(3:4);
    set(grp_plots(idx - 1),'Position',tmp);
    set(get(grp_plots(idx - size(opt.interval, 1)),'Ylabel'), 'Visible','on', 'String', {opt.class{i};''}, ...
        'FontWeight', 'normal', 'FontSize', 12);
end
end