function [cnt, mrk, sbjData] = load_online_data_T9wharp(VPDir, varargin)

[cnt, mrk, foo, test_epo_idx, sbjData] = load_data_T9wharp(VPDir, varargin{:});
mrk = mrk_selectEvents(mrk, test_epo_idx);

