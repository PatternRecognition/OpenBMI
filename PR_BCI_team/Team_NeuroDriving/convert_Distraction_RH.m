clear all; clc; close all;

subjectList= {'HSRYU_20180406_distraction','JHHWANG_20180407_distraction','GSPARK_20180413_distraction','DNJO_20180414_distraction'};

grd= sprintf(['_,_,_,FP1,_,_,_,FP2,_,_,_\n', ...
    '_,_,_,AF3,_,_,_,AF4,_,_,_\n'...
    '_,_,F5,F3,F1,Fz,F2,F4,F6,_,_\n', ...
    'FT9,FT7,FC5,FC3,FC1,_,FC2,FC4,FC6,FT8,FT10\n', ...
    '_,T7,C5,C3,C1,Cz,C2,C4,C6,T8,_\n', ...
    'TP9,TP7,CP5,CP3,CP1,CPz,CP2,CP4,CP6,TP8,TP10\n', ...
    '_,P7,P5,P3,P1,Pz,P2,P4,P6,P8,_\n'...
    '_,_,PO7,PO3,POz,PO4,PO8,_,_,_,_\n'...
    'legend,_,_,PO9,O1,Oz,O2,PO10,_,_,scale\n']);

file_basic = 'D:\ICMI\Data\Distraction\';
saveFile = 'D:\ICMI\Data\Distraction\MATLAB';

for s = 1:length(subjectList)
    file = strcat(file_basic, '\', subjectList{s});
    
    try
        hdr= eegfile_readBVheader(file);
    catch
        fprintf('%s not found.\n', file);
    end
    
    [cnt, mrk_orig]= eegfile_loadBV(file, 'fs',250, ...
        'clab',{'not','EMG*'});

    cnt.title= file;
    
    mrk= marker_Distraction_Define(mrk_orig);
    
    % Elimination of EEG data prior to start of experiment and posterior to
    % end of experiment
    cnt.x = cnt.x(mrk.misc.pos(1) : mrk.misc.pos(end), :);
    
    % setting the kss marker and synchronizing the start and end point
    mrk.pos = mrk.pos - mrk.misc.pos(1, 1) + 1;
    mrk.misc.pos = mrk.misc.pos - mrk.misc.pos(1, 1) + 1;
    
    mnt= getElectrodePositions(cnt.clab);
    mnt= mnt_setGrid(mnt, grd);
    mnt= mnt_excenterNonEEGchans(mnt,'E*');
    
    fs_orig= mrk_orig.fs;
    var_list= {'fs_orig',fs_orig, 'mrk_orig',mrk_orig, 'hdr',hdr};
    
    eegfile_saveMatlab(strcat(saveFile, '\', subjectList{s}), cnt, mrk, mnt, ...
        'channelwise',1, ...
        'format','double', ...
        'resolution', NaN);
end

