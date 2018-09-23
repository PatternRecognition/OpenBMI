function grp_plots = vis_timeAveragePlot(SMT, varargin)
% Description:
%
% Input:
%       plts : group of axes 
%       SMT  : epoched EEG data
% Options       :
%       'Channels',
%       'Class',
%       'Rnage',
%       'Baseline',
%       'Envelope',
% Ouput:
%       grp_plots   : cell array of axes
% Example:
%       Options = {'Channels', {'Cz', 'Oz'};
%                   'Class', {'target'; 'non_target'};
%                   'Range', [100 200];
%                   'Baseline', [-200 0];
%                   'Envelope', false};
%       axis = vis_timeAveragePlot(SMT, Options);
%
% Created by Hong Kyung, Kim
% hk_kim@korea.ac.kr


opt = [varargin{:}];
if ~isstruct(opt) && iscell(opt)
    opt = opt_cellToStruct(opt);
end

faceColor = [{[0.8 0.8 0.8]};{[0.5 0.5 0.5]}];
%% Options
if ~isfield(opt, 'channels') opt.channels = {SMT.chan{1}}; end
if ~isfield(opt, 'class') opt.class = {SMT.class{1,2}}; end
if ~isfield(opt, 'baseline') opt.baseline = [SMT.ival(1) SMT.ival(1)]; end
if ~isfield(opt, 'selectTime') opt.selectTime = [SMT.ival(1) SMT.ival(end)]; end
if ~isfield(opt, 'interval') opt.interval = [opt.SelectTime(1) opt.SelectTime(end)]; end
if ~isfield(opt, 'patch') opt.patch = 'off'; end
if ~isfield(opt, 'plots') opt.plots = gca; end

grp_plots = opt.plots;

%% TODO: time_range 변경
% 1. class 선택
% 2. channel 선택
% 3. ???????

%% 임시
SMT = prep_selectChannels(SMT, {'Name', opt.channels});
%%
if isfield(opt, 'timerange') && ~isempty(opt.timerange)
    time_range = opt.timerange;
else
    time_range = minmax(reshape(SMT.x, 1, []));
end

%% opt.class 순서대로
[~, selected_class_order] = ismember(opt.class, SMT.class(:,2));

idx = 1;
for ch_num = find(ismember(SMT.chan, opt.channels)) 
    plot(grp_plots(idx), SMT.ival, SMT.x(:,selected_class_order',ch_num),'LineWidth',2); hold on;
    legend(grp_plots(idx),opt.class, 'Interpreter', 'none', 'AutoUpdate', 'off'); % TODO: 2014b 호환되지 않음 'AutoUpdate'
    
    grid(grp_plots(idx), 'on');
    ylim(grp_plots(idx), time_range);
    xlim(grp_plots(idx), [SMT.ival(1) SMT.ival(end)]);
    if isequal(opt.patch, 'on')
        % Baseline patch
        base_patch = min(abs(time_range))*0.1;
        
        patch(grp_plots(idx), 'XData', [opt.baseline(1)  opt.baseline(2) opt.baseline(2) opt.baseline(1)], ...
            'YData', [-base_patch -base_patch base_patch base_patch],...
            'FaceColor', 'k', 'FaceAlpha', 0.7, 'EdgeAlpha', 0,'faceOffsetBias', -11);
        % Ival patch
        for ival = 1:size(opt.interval,1)
            patch(grp_plots(idx), 'XData', [opt.interval(ival,1) opt.interval(ival,2) opt.interval(ival,2) opt.interval(ival,1)],...
                'YData', [time_range(1) time_range(1) time_range(2) time_range(2)],...
                'FaceColor', faceColor{mod(ival,2)+1}, 'FaceAlpha', 0.3, 'EdgeAlpha', 0,'faceOffsetBias', -11);
        end
        
        tmp = get(grp_plots(idx), 'Children');
        set(grp_plots(idx), 'Children', flip(tmp));
    end
    ylabel(grp_plots(idx), SMT.chan{ch_num}, 'Rotation', 90, 'FontWeight', 'normal', 'FontSize', 12);
    hold(grp_plots(idx), 'off');
    idx = idx + 1;
end
end
