% EEG_MAT_DIR= 'C:\uni\lab rotation\bbciMat\';
% 
% files = {
%  
% 'Thomas_09_10_23/Thomas_vis_count', 
% 'Thomas_09_10_23/Thomas_tact_count',
% 'sophie_09_10_30/sophie_vis_count',
%  'sophie_09_10_30/sophie_tact_count',
% 'chris_09_11_17/chris_vis_count',
% 'chris_09_11_17/chris_tact_count',
% 'nico_09_11_12/nico_vis_count',
% 'nico_09_11_12/nico_tact_count',
% 'rithwick_09_11_05/rithwick_vis_count',   
% 'rithwick_09_11_05/rithwick_tact_count'
%         };
% clear loss
% for f= 1:size(files,1),    
%     [cnt, mrk, mnt]= eegfile_loadMatlab(files(f));
% 
% 
%     disp_ival= [-100 1200];
% 
%     'blah';
%     %  rejection of broad-band artifacts
%     [mk, rClab]= reject_varEventsAndChannels(cnt, mrk, disp_ival, ...
%                                               'visualize', 0);
%     mrk= mk;
%     epo= cntToEpo(cnt, mrk, [-100 1200]);
%     epo= proc_baseline(epo, [-100 0]);
%     epo= proc_detrend(epo);
%     epo= proc_selectIval(epo, [0 1000]);
%     epo_r= proc_r_square_signed(epo);
%     ival= select_time_intervals(epo_r, 'visualize',0, 'visu_scalps',1, ...
%                                'nIvals',7, 'sort',1)
% 
% 
%     opt_xv= struct('loss', 'classwiseNormalized')
%     for ii= 1:size(ival,1),    
%      ind  = (f-1)*6 + ii;
%      loss(ind,1)= f;
%      loss(ind,2)= ival(ii,1);
%      loss(ind,3)= ival(ii,2);
%     %  ['interval:' num2str(ival(ii,1)) ' ' num2str(ival(ii,2))]
%      fv= proc_selectIval(epo, ival(ii,:));
%      fv= proc_meanAcrossTime(fv);
%     %  'LDA'
%      loss(ind,4)= xvalidation(fv, 'LDA', opt_xv);
%     %  'Fisher'
%      loss(ind,5)= xvalidation(fv, 'FisherDiscriminant', opt_xv);
%     %  'Shrink'
%       loss(ind,6)= xvalidation(fv, 'RLDAshrink', opt_xv)
%     end
% end
% 
% save('C:\uni\lab rotation\bbciMat\classResults.mat', 'loss', '-ASCII')
% save('C:\uni\lab rotation\bbciMat\fileIndex.mat', 'files')

loss= load('C:\uni\lab rotation\bbciMat\classResults.mat', '-ASCII')
fileIndex = load('C:\uni\lab rotation\bbciMat\fileIndex.mat')

close all
clear xLab
for i = 1:6,
   xLab(i) =  {[num2str(loss(i,2)) '-' num2str(loss(i,3)) ]}
end


for i = 1:size(fileIndex.files,1)-1,
    figure(i)
    ind = (i-1) * 6
    bar(loss(ind + 1: ind + 6,4:6))
    set(gca,'xticklabel',xLab)
    legend({'LDA';'Fisher Disc';'LDA Shrink'})
    title(fileIndex.files(i,1))
    ylim([0 0.5])
end

clear bestVisTime
clear bestTactTime
clear vis
clear tact
clear bestVisMethod
clear bestTactMethod
ind = 1
for i = 1:2:size(fileIndex.files,1),
    indVis = (i-1) * 6;
    indTact = i * 6;
    bestVisTime(ind,:) = min(loss(indVis+ 1: indVis + 6,4:6)');
    bestTactTime(ind,:) = min(loss(indTact+ 1: indTact + 6,4:6)');
    bestVisMethod(ind,:) = min(loss(indVis+ 1: indVis + 6,4:6));
    bestTactMethod(ind,:) = min(loss(indTact+ 1: indTact + 6,4:6));
    %vis(ind:ind+6,1:3) = 
    loss(indVis+ 1: indVis + 6,4:6)
    %tact(ind:ind+6,:) = loss(indTact+ 1: indTact + 6,4:6);
%     set(gca,'xticklabel',xLab)
%     legend({'LDA';'Fisher Disc';'LDA Shrink'})
%     title(fileIndex.files(i,1))
%     ylim([0 0.5])
    ind = ind + 1;
end


a = mean(bestVisTime);
b = mean(bestTactTime);
c = mean(bestVisMethod);
d = mean(bestTactMethod);

close all
figure(1)
plot(bestVisTime','o-')
hold all
plot(mean(bestVisTime), 'x-')
set(gca,'xticklabel',xLab)
ylim([0 0.5])
title('best vis time')
hold off

figure(2)
plot(bestTactTime','o-')
hold all
plot(mean(bestTactTime), 'x-')
set(gca,'xticklabel',xLab)
ylim([0 0.5])
title('best tact time')
hold off


figure(3)
plot(bestVisMethod','o-')
hold all
plot(mean(bestVisMethod), 'x-')
set(gca,'xticklabel',xLab)
ylim([0 0.5])
title('best vis Method')
hold off

figure(4)
plot(bestTactMethod','o-')
hold all
plot(mean(bestTactMethod), 'x-')
set(gca,'xticklabel',xLab)
ylim([0 0.5])
title('best tact Method')
hold off


figure(5)
bar({a; b; c; d;})

