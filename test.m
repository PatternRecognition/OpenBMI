clear all; clc;
%% subject info
subjects = {'mhlee'};
sub_num = {0};
datapath = 'D:\대학원 자료\실험 자료\Data';
session = 'session1';
files = {'exp1_passive', 'exp1_imagery', 'exp2_cue',...
    'exp1_passive', 'exp1_imagery', 'exp2_ncue'};

band_freq = [0.4 40];
notch_freq = [8 12];

selChans = {'FC1','FC2', 'C3', 'Cz','C4','CP1','CP2'};
selChans = {'F3', 'Fz','F4','FC5','FC1','FCz','FC2','FC6','T7','C3','C1','Cz','C2','C4','T8','CP5','CP1','CPz','CP2','CP6','P7','P3','P1','Pz','P2','P4','P8','O1','Oz','O2'};

subject = 1;
filenum = 1;

file = fullfile(datapath, sprintf('subject%d_%s',sub_num{subject}, subjects{subject}), session, files{filenum});
field={'x','t','fs','y_dec','y_logic','y_class','class', 'chan'};
marker={'1', 'Gaze'};
[EEG.data, EEG.marker, EEG.info]=Load_EEG(file,{'device','brainVision';'marker', marker});
cnt=opt_eegStruct({EEG.data, EEG.marker, EEG.info}, field);
smt=prep_segmentation(cnt, {'interval', [0 200]});
selectedSmt=prep_selectChannels(smt, 1:16);
selectedSmt=prep_selectChannels(smt, selChans);
selectedSmt=prep_selectChannels(smt, {'Name', selChans});
selectedCnt=prep_selectChannels(cnt, {'Name', selChans});
selectedCnt=prep_selectChannels(cnt, {'Name', selChans;'Index', 1:32});