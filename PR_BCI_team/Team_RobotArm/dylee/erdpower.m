%% erd power

clc; close all; clear all;
%% file
dd='C:\Users\Doyeunlee\Desktop\Analysis\rawdata\convert\';
% reaching
filelist={'eslee_reaching_MI','jmlee_reaching_MI','dslim_reaching_MI'};
% filelist={'eslee_reaching_realMove','jmlee_reaching_realMove','dslim_reaching_realMove'};

% multigrasp
% filelist={'eslee_multigrasp_MI','jmlee_multigrasp_MI','dslim_multigrasp_MI'};
% filelist={'eslee_multigrasp_realMove','jmlee_multigrasp_realMove','dslim_multigrasp_realMove'};

% twist
% filelist={'eslee_twist_MI','jmlee_twist_MI','dslim_twist_MI'};
% filelist={'eslee_twist_realMove','jmlee_twist_realMove','dslim_twist_realMove'};

for i=length(filelist)
    [cnt, mrk, mnt] = eegfile_loadMatlab([dd filelist{i}]);
    
    ival = [-500 3000];
    epo = cntToEpo(cnt, mrk, ival);
    
    fv = epo;
    fv = proc_laplacian(fv, 'small', 'lap');
    
    fv = proc_selectChannels(fv,5);
    
    spec = proc_spectrum(fv, [5 35]);
    
    power=reshape(spec.x,31,600);
    power_mean=mean(power,2);
    frq=5:35;
    
    figure(i);
    ylim([-4 12]);
    xlabel('Hz');
    ylabel('dB');
    
    hold on;
    for k=1:40
        plot(frq,power(:,k),'Color',[175 175 175]/255)
    end
    plot(frq,power_mean,'Color','Red')
    hold off;
    

    
end
