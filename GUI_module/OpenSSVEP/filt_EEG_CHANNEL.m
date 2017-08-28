function [EEG, OTHERS, STRMATRIX] = filt_EEG_CHANNEL(data_chan)
% Description:
% 입력된 채널 정보와 표준 128채널의 정보를 비교하여 위치를 찾아주고,
% 입력된 채널의 index 정보를 함께 보여주는 역할을 하는 함수
%
% Input: (1xN size, cell type(?))
% data_chan - 현재 채널의 데이터 1xN의 형태
%
% Output
% EEG       - 128채널에 존재하는 채널정보와 그 index
% OTHERS    - 128채널에 존재하지 않는 채널 정보와 그 index
% STRNATRIX - Scalp plot에 기반하여 위치시킨 string 정보

%% Initialization
EEG = {};
OTHERS = {};
chan = {...
    '','','','Fp1','','Fpz','', 'Fp2','','','';
    '','','','','AFp1','','AFp2', '','','','';
    'F9','AF7','', 'AF3','','AFz','','AF4', '','AF8','F10';
    '', '','AFF5h','','AFF1h', '', 'AFF2h', '', 'AFF6h','','';
    '','F7', 'F5', 'F3', 'F1', 'Fz', 'F2', 'F4', 'F6', 'F8','';
    'FFT9h','FFT7h','FFC5h','FFC3h','FFC1h','','FFC2h','FFC4h','FFC6h','FFT8h','FFT10h';
    'FT9', 'FT7', 'FC5', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'FC6', 'FT8', 'FT10';
    'FTT9h','FTT7h','FCC5h','FCC3h','FCC1h','','FCC2h','FCC4h','FCC6h','FTT8h','FTT10h';
    '', 'T7', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 'T8','';
    '', 'TTP7h', 'CCP5h', 'CCP3h', 'CCP1h', '', 'CCP2h', 'CCP5h', 'CCP6h', 'TTP8h', '';
    'TP9', 'TP7', 'CP5', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6', 'TP8', 'TP10';
    'TPP9h', 'TPP7h', 'CPP5h', 'CPP3h', 'CPP1h', '', 'CPP2h', 'CPP4h', 'CPP6h', 'TPP8h', 'TPP10h';
    'P9','P7', 'P5', 'P3', 'P1', 'Pz', 'P2', 'P4', 'P6', 'P8','P10';
    'PPO9h', '', 'PPO5h', '', 'PPO1h', '', 'PPO2h', '', 'PPO6h', '', 'PPO10h';
    '','PO7','','PO3', '', 'POz', '', 'PO4','','PO8','';
    '','PO9','POO9h','O1','POO1','','POO2','O2','POO10h','PO10','';
    '','','','l1', 'Ol1h', 'Oz', 'Olh2', 'l2','','','';
    '','','','', '', 'lz', '', '','','',''};
STRMATRIX = cell(size(chan,1)+1, size(chan,2));
j = size(chan,2);

%% Filtering channels
for i = 1: size(data_chan, 2)
    [row, col]= find(strcmp(chan,data_chan{i}));
    if(isempty(row)&&isempty(col))
        STRMATRIX{size(STRMATRIX,1), j} = sprintf('%s (%d)',data_chan{i},i);
        j = j-1;
        OTHERS = vertcat(OTHERS, {i,data_chan{i}});
    else
        STRMATRIX{row, col} = sprintf('%s (%d)',data_chan{i},i);
        if isempty(EEG)
            EEG = {i, data_chan{i}};
        else
            EEG = vertcat(EEG, {i, data_chan{i}});
        end
    end
end

EEG=EEG';
OTHERS=OTHERS';