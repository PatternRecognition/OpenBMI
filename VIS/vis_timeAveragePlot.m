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
if ~isfield(opt, 'Channels') opt.Channels = {SMT.chan{1}}; end
if ~isfield(opt, 'Class') opt.Class = {SMT.class{1,2}}; end
if ~isfield(opt, 'Baseline') opt.Baseline = [SMT.ival(1) SMT.ival(1)]; end
if ~isfield(opt, 'SelectTime') opt.SelectTime = [SMT.ival(1) SMT.ival(end)]; end
if ~isfield(opt, 'Interval') opt.Interval = [opt.SelectTime(1) opt.SelectTime(end)]; end
if ~isfield(opt, 'Patch') opt.Patch = 'off'; end
if ~isfield(opt, 'Plots') opt.Plots = gca; end

grp_plots = opt.Plots;

%% TODO: time_range 변경
% 1. class 선택
% 2. channel 선택
% 3. ???????

%% 임시
SMT = prep_selectChannels(SMT, {'Name', opt.Channels});
%%
if isfield(opt, 'TimeRange') && ~isempty(opt.TimeRange)
    time_range = opt.TimeRange;
else
    time_range = minmax(reshape(SMT.x, 1, []));
end

idx = 1;
for ch_num = find(ismember(SMT.chan, opt.Channels)) 
    plot(grp_plots(idx), SMT.ival, SMT.x(:,:,ch_num),'LineWidth',2); hold on;
    legend(grp_plots(idx),opt.Class, 'Interpreter', 'none', 'AutoUpdate', 'off');
    %         set({'color'}, co(1:size(avgSMT.class, 1)));
    
    grid(grp_plots(idx), 'on');
    ylim(grp_plots(idx), time_range);
    xlim(grp_plots(idx), [SMT.ival(1) SMT.ival(end)]);
    if isequal(opt.Patch, 'on')
        % baselin patch
        base_patch = min(abs(time_range))*0.05;
        
        patch(grp_plots(idx), 'XData', [opt.Baseline(1)  opt.Baseline(2) opt.Baseline(2) opt.Baseline(1)], ...
            'YData', [-base_patch -base_patch base_patch base_patch],...
            'FaceColor', 'k',...
            'FaceAlpha', 0.7, 'EdgeAlpha', 0,'faceOffsetBias', -11);
        % ival patch
        for ival = 1:size(opt.Interval,1)
            patch(grp_plots(idx), 'XData', [opt.Interval(ival,1) opt.Interval(ival,2) opt.Interval(ival,2) opt.Interval(ival,1)],...
                'YData', [time_range(1) time_range(1) time_range(2) time_range(2)],...
                'FaceColor', faceColor{mod(ival,2)+1},...
                'FaceAlpha', 0.3, 'EdgeAlpha', 0,'faceOffsetBias', -11);
        end
        
        tmp = get(grp_plots(idx), 'Children');
        set(grp_plots(idx), 'Children', flip(tmp));
    end
    ylabel(grp_plots(idx), SMT.chan{ch_num}, 'Rotation', 90, 'FontWeight', 'normal', 'FontSize', 12);
    hold(grp_plots(idx), 'off');
    idx = idx + 1;
end
end
