function [cnt, mrk, sbjData] = load_calibration_data_RSVP(VPDir, varargin)

% opt= propertylist2struct(varargin{:});
% opt= set_defaults(opt ...
%     );
[cnt, mrk, sbjData] = load_data_RSVP(VPDir, varargin{:});
% select only the calibration phase data
calibration_mode_indices = find(mrk.mode(1,:));
mrk = mrk_selectEvents(mrk, calibration_mode_indices);
