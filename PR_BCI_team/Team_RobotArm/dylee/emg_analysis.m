% Extracting EMG signals
clear all; clc; close all;

dd='C:\Users\Doyeunlee\Desktop\Journal_dylee\0_RawData\sub01\';
filelist={'session1_sub1_multigrasp_realMove'};
for ff=1:length(filelist)
    file=filelist{ff};
    opt=[];
    try
        hdr=eegfile_readBVheader([dd '\' file]);
    catch
        fprintf('%s/%s not found.\n', dd, file);
        continue;
    end
    
    % filtering with Chev filter
    Wps= [42 49]/hdr.fs*2;
    [n, Ws]= cheb2ord(Wps(1), Wps(2),3, 40);
    [filt.b, filt.a]= cheby2(n, 50, Ws);
    
    [cnt, mrk_orig]=eegfile_loadBV([dd '\' file],'filt', filt, 'fs', 1000);
    
    % Save the file
    cnt.title=['C:\Users\Doyeunlee\Desktop\Analysis\01_Raw data\' file '_EMG_1'];
    mrk=WAM_20170814_Imag_Arrow(mrk_orig);
    mnt=getElectrodePositions(cnt.clab);
    fs_orig=mrk_orig.fs;
    var_list={'fs_orig', fs_orig, 'mrk_orig', mrk_orig, 'hdr', hdr};
    
    eegfile_saveMatlab(cnt.title,cnt,mrk,mnt, ...
        'channelwise',1, ...
        'format', 'int16',...
        'resolution', NaN);
end

disp('All EMG Data Converting was Done!');
    
    
