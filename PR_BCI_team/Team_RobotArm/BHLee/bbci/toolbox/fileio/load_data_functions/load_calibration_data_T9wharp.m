function [cnt, mrk, sbjData] = load_calibration_data_T9wharp(VPDir, varargin)

[cnt, mrk, train_epo_idx, foo, sbjData] = load_data_T9wharp(VPDir, varargin{:});
mrk = mrk_selectEvents(mrk, train_epo_idx);

