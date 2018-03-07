function grp_plots = vis_topoPlot(plts, SMT, varargin)
% Description:
%
%
%
%
%
%
%
PREVENT_OVERFLOW = 5000;
%% Options

switch nargin
    case 2
        opt = [];
    case 3
        opt = varargin{:};
        if ~isstruct(opt)
            opt = opt_cellToStruct(opt);
        end
end

%% Creating Montage
MNT = opt_getMontage(SMT);
output_str = [];

if ~isequal(MNT.chan, SMT.chan)
    if length(SMT.chan) > length(MNT.chan)
        tmp = SMT.chan(~ismember(SMT.chan, MNT.chan));
        output_str = tmp{1};
        for i = 2:length(tmp)
            output_str = [output_str sprintf(', %s',tmp{i})];
        end
        output_str = {''; [output_str, ' are missed']};
    end
    tmp = zeros(1,length(MNT.x));
    for i = 1:length(MNT.chan)
        tmp(i) = find(ismember(SMT.chan, MNT.chan{i}));
    end
    SMT.x = SMT.x(:,:,tmp);
    SMT.chan = SMT.chan(tmp);
    clear tmp;
end

if ~isfield(opt, 'Colormap') opt.Colormap = 'parula'; end
if ~isfield(opt, 'Quality') opt.Quality = 'high'; end
if ~isfield(opt, 'Class') opt.Class = {SMT.class{1:2,2}}; end
if ~isfield(opt, 'Baseline') opt.Baseline = [SMT.ival(1) SMT.ival(1)]; end
if ~isfield(opt, 'SelectTime') opt.SelectTime = [SMT.ival(1) SMT.ival(end)]; end
if ~isfield(opt, 'Interval') opt.Interval = [opt.SelectTime(1) opt.SelectTime(end)]; end
if ~isfield(opt, 'Channels') opt.Channels = {SMT.chan{1:5}}; end
if ~isfield(opt, 'Range') opt.Range = 'sym'; end

grp_plots = plts;

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

if size(SMT.x, 2) > PREVENT_OVERFLOW
    divis = divisors(size(SMT.x, 2));
    divis = divis(divis < PREVENT_OVERFLOW);
    divis = divis(end);
    SMT_size = size(SMT.x, 2);
    for i = 1:divis:SMT_size
        tmpSMT = prep_selectTrials(SMT, {'Index', i:i+divis-1});
        tmpSMT = prep_selectClass(tmpSMT ,{'class',opt.Class});
        %         tmpSMT = prep_selectClass(tmpSMT ,{'class',opt.Class});
        if opt.Envelope
            tmpSMT = prep_envelope(tmpSMT);
        end
        tmpSMT = prep_baseline(tmpSMT, {'Time', opt.Baseline});
        SMT.x(:,i:i+divis-1,:) = tmpSMT.x;
    end
    
    clear tmpSMT;
else
    SMT = prep_selectClass(SMT,{'class',opt.Class});
    if isfield(opt, 'Envelope') && opt.Envelope
        SMT = prep_envelope(SMT);
    end
    SMT = prep_baseline(SMT, {'Time', opt.Baseline});
end
SMT = prep_selectTime(SMT, {'Time', opt.SelectTime});
SMT = prep_average(SMT);

idx = 1;
for i = 1: size(opt.Class, 1)
    ivalSegment = size(opt.Interval,1);
    minmax = [];
    for seg = 1: ivalSegment
        SMTintervalstart = find(SMT.ival == opt.Interval(seg,1));
        SMTintervalEnd = find(SMT.ival == opt.Interval(seg,2));
        ivalSMT = squeeze(SMT.x(SMTintervalstart:SMTintervalEnd,i,:));
        w{i, seg} = mean(ivalSMT,1);
        minmax = [minmax; min(w{i,seg}(:)), max(w{i,seg}(:))];
    end
    %% range_options
    if ~isfloat(opt.Range)
        switch opt.Range
            case 'sym'
                p_range = [-max(abs(max(minmax(:,2)))), max(abs(max(minmax(:,2))))];
            case '0tomax'
                p_range = [0.0001*diff([min(minmax(:,1)) max(minmax(:,2))]), max(minmax(:,2))];
            case 'minto0'
                p_range = [min(minmax(:,1)), 0.0001*diff([min(minmax(:,1)) max(minmax(:,2))])];
            case 'mintomax'
                p_range = [min(minmax(:,1)) max(minmax(:,2))];
            case 'mean'
                p_range = [mean(minmax(:,1)) mean(minmax(:,2))];
        end
    else
        p_range = opt.Range;
    end
    if diff(p_range)==0
        p_range(2)= p_range(2)+eps;
    end
    %% Draw
    for seg = 1: size(opt.Interval, 1)
        plot_scalp(grp_plots{idx}, w{i, seg}, MNT, p_range, resol);
        xlabel(grp_plots{idx}, sprintf('[%d - %d] ms',opt.Interval(seg,:)), 'FontWeight', 'normal');
        idx = idx + 1;
    end
    % Keeping scalp size;
    last_position = get(grp_plots{idx - 1}, 'Position');
    colorbar(grp_plots{idx - 1}, 'vert');
    tmp = get(grp_plots{idx - 1}, 'Position');
    tmp(3:4) = last_position(3:4);
    set(grp_plots{idx - 1},'Position',tmp);
    set(get(grp_plots{idx - size(opt.Interval, 1)},'Ylabel'), 'Visible','on', 'String', {opt.Class{i};''}, ...
        'Interpreter', 'none', 'FontWeight', 'normal', 'FontSize', 12);
end
end