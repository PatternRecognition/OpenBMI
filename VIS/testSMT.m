%% Data
clear all; clc;
%% subject info
subjects = {'mhlee', 'hjwon', 'yelee', 'hkkim', 'dsjung', 'yjkim'};
sub_num = {1, 2, 3, 4, 5, 6};
date = {'20180104', '20180104', '20180105', '20180105', '20180106', '20180109'};
datapath = 'D:\대학원 자료\Conference\SmallSti(BCI_Meeting)\Data\';

subject = 6;

files = {sprintf('%s_normal_gaze_training_%s', date{subject}, subjects{subject}),...
    sprintf('%s_normal_gaze_test_%s_02', date{subject}, subjects{subject}),...
    sprintf('%s_dot_gaze_training_%s_02', date{subject}, subjects{subject}),...
    sprintf('%s_dot_gaze_test_%s', date{subject}, subjects{subject}),...
    sprintf('%s_dot_pitch_training_%s', date{subject}, subjects{subject}),...
    sprintf('%s_dot_pitch_test_%s', date{subject}, subjects{subject})};

for filenum = 1:length(files)
    file = fullfile(datapath, sprintf('S%d_%s',sub_num{subject}, subjects{subject}), files{filenum});
    
    field={'x','t','fs','y_dec','y_logic','y_class','class', 'chan'};
    marker={'1', 'target';'2','non_target'};
    % marker={'1', 'right';'2','left'};
    [EEG.data, EEG.marker, EEG.info]=Load_EEG(file,{'device','brainVision';'marker', marker;'fs',100});
    CNT=opt_eegStruct({EEG.data, EEG.marker, EEG.info}, field);
    CNT=prep_filter(CNT, {'frequency', [0.5 40]});
    SMT=prep_segmentation(CNT, {'interval', [-200 800]});
    % CNT=prep_filter(CNT, {'frequency', [8 13]});
    % SMT=prep_segmentation(CNT, {'interval', [-490 4000]});
    %% Code
    ival = [200 300; 300 400; 400 500; 500 600; 600 700];
    chan = {'Cz', 'Oz'};
    class = {'target'; 'non_target'};
    TimePlot = 'on'; TopoPlot = 'on'; ErspPlot = 'off'; ErdPlot = 'off';
    range = [-2 2]; cm = 'Parula'; baseline = [-200 0]; patch = 'off'; quality = 'high';
    timerange = [-5 5];
    
    vis_plotCont2(SMT, {'Interval', ival; 'Channels',chan; 'Class',class;...
        'TimePlot', TimePlot; 'TopoPlot', TopoPlot; 'ErspPlot', ErspPlot;...
        'ErdPlot', ErdPlot; 'Range', range; 'Baseline', baseline;...
        'Colormap', cm; 'Patch', patch; 'Quality', quality; 'TimeRange', timerange});
    saveas(gcf, fullfile(datapath, sprintf('S%d_%s',sub_num{subject}, subjects{subject}), 'figure', files{filenum}),'jpg');
    close(gcf);
end