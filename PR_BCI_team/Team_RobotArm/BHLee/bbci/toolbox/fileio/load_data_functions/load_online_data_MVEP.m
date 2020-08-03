function [cnt, mrk, sbjData] = load_online_data_MVEP(VPDir, varargin)

% opt= propertylist2struct(varargin{:});
% opt= set_defaults(opt ...
%     );

[cnt, mrk, sbjData] = load_data_MVEP(VPDir, varargin{:});
% select only the online spelling phase data
online_mode_indices = find(sum(mrk.mode(2:3,:)));
mrk = mrk_selectEvents(mrk, online_mode_indices);
