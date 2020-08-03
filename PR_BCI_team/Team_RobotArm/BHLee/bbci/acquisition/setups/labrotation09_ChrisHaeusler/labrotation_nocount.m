EEG_MAT_DIR= 'C:\uni\lab rotation\bbciMat\';
LATEX_DIR= 'C:\uni\lab rotation\images\';
results_file = ''

NO_COUNT = 0
DO_CLASSIFY = 0
DO_IMAGES = 1
clear subject results_file class_results
close all

base_ival = [-100 0];
epo_ival = [-100 700];
disp_ival =[50 700];
crit_ival= [100 700];
    
num_ival = 3
if NO_COUNT == 1
    results_file = 'classResultsNoCount.mat'
    class_results = 'NoCount'
    subject(1).name='VPnh_vis_nocount';
    subject(1).path='Thomas_09_10_23/Thomas_vis_nocount'; 
    subject(2).name='VPnh_tact_nocount';
    subject(2).path='Thomas_09_10_23/Thomas_tact_nocount';
    subject(3).name='VPmk_vis_nocount';
    subject(3).path='sophie_09_10_30/sophie_vis_nocount';
    subject(4).name='VPmk_tact_nocount';
    subject(4).path='sophie_09_10_30/sophie_tact_nocount';
    subject(5).name='VPgao_vis_nocount';
    subject(5).path='chris_09_11_17/chris_vis_nocount';
    subject(6).name='VPgao_tact_nocount';
    subject(6).path='chris_09_11_17/chris_tact_nocount';
    subject(7).name='VPiac_vis_nocount';
    subject(7).path='nico_09_11_12/nico_vis_nocount';
    subject(8).name='VPiac_tact_nocount';
    subject(8).path='nico_09_11_12/nico_tact_nocount';
    subject(9).name='rithwick_vis_nocount';
    subject(9).path='rithwick_09_11_05/rithwick_vis_nocount';   
    subject(10).name='rithwick_tact_nocount';
    subject(10).path='rithwick_09_11_05/rithwick_tact_nocount';
else
    class_results = 'Count'
    results_file = 'classResults.mat'
    subject(1).name='VPnh_vis_count';
    subject(1).path='Thomas_09_10_23/Thomas_vis_count'; 
    subject(2).name='VPnh_tact_count';
    subject(2).path='Thomas_09_10_23/Thomas_tact_count';
    subject(3).name='VPmk_vis_count';
    subject(3).path='sophie_09_10_30/sophie_vis_count';
    subject(4).name='VPmk_tact_count';
    subject(4).path='sophie_09_10_30/sophie_tact_count';
    subject(5).name='VPgao_vis_count';
    subject(5).path='chris_09_11_17/chris_vis_count';
    subject(6).name='VPgao_tact_count';
    subject(6).path='chris_09_11_17/chris_tact_count';
    subject(7).name='VPiac_vis_count';
    subject(7).path='nico_09_11_12/nico_vis_count';
    subject(8).name='VPiac_tact_count';
    subject(8).path='nico_09_11_12/nico_tact_count';
    subject(9).name='rithwick_vis_count';
    subject(9).path='rithwick_09_11_05/rithwick_vis_count';   
    subject(10).name='rithwick_tact_count';
    subject(10).path='rithwick_09_11_05/rithwick_tact_count'; 
    
end

if DO_CLASSIFY == 1
    clear loss
    for f= 1:length(subject),    
        [cnt, mrk, mnt]= eegfile_loadMatlab(subject(f).path);


        [mk, rClab]= reject_varEventsAndChannels(cnt, mrk, disp_ival, ...
                                          'visualize', 0);
        mrk= mk;

        epo= cntToEpo(cnt, mrk,epo_ival);
        epo= proc_baseline(epo, base_ival);
        %epo= proc_detrend(epo);
        epo_crit= proc_selectIval(epo, crit_ival);
        iArte= find_artifacts(epo_crit, 'EOG*', struct('maxmin', 150));
        epo= proc_selectEpochs(epo, 'not',iArte);
        epo= proc_selectIval(epo, disp_ival);
        epo_r= proc_r_square_signed(epo);
        ival= select_time_intervals(epo_r, 'visualize',0, 'visu_scalps',1, ...
                                   'nIvals',num_ival, 'sort',1)


        opt_xv= struct('loss', 'classwiseNormalized')
        for ii= 1:num_ival,    
         ind  = (f-1)*num_ival + ii;
         loss(ind,1)= f;
         loss(ind,2)= ival(ii,1);
         loss(ind,3)= ival(ii,2);
         fv= proc_selectIval(epo, ival(ii,:));
         fv= proc_meanAcrossTime(fv);
        %  'LDA'
         loss(ind,4)= xvalidation(fv, 'LDA', opt_xv);
        %  'Fisher'
         loss(ind,5)= xvalidation(fv, 'FisherDiscriminant', opt_xv);
        %  'Shrink'
          loss(ind,6)= xvalidation(fv, 'RLDAshrink', opt_xv)
        end
    end

    
    save([EEG_MAT_DIR results_file], 'loss', '-ASCII')
    
end

if DO_IMAGES == 1
    loss= load([EEG_MAT_DIR results_file], '-ASCII')
    
    close all
    
   

    methods = {'LDA';'Fisher Disc';'LDA Shrink';}

    ii = 1
    for i = 1:2:length(subject),
        h=figure(ii)

        subplot(211)


        ind = (i) * num_ival
        
        clear xLab
        for int = 1 : num_ival,
            xLab(int) =  {[num2str(loss(int + ind,2)) '-' num2str(loss(int + ind,3)) ]}
        end
        
        bar(loss(ind + 1: ind + num_ival,4:6))
        set(gca,'xticklabel',xLab)
        legend(methods, 'best')
        ylim([0 0.5])
        ylabel('Error %')
        xlabel('Time Window (ms after stimulus)')        
        title('Visual Classification Error')

        subplot(212)

        ind = (i-1) * num_ival
        
        clear xLab
        for int = 1 : num_ival,
            xLab(int) =  {[num2str(loss(int + ind,2)) '-' num2str(loss(int + ind,3)) ]}
        end
        bar(loss(ind + 1: ind + num_ival,4:6))
        set(gca,'xticklabel',xLab)
        legend(methods, 'best')
        ylim([0 0.5])
        ylabel('Error %')
        xlabel('Time Window (ms after stimulus)')
        title('Tactile Classification Error')

        savename=fullfile(LATEX_DIR, [char(subject(i).name) '_class_error'])
        set(gcf,'paperunits','centimeters')
        set(gcf,'papersize',[22,20])
        saveas(h,savename,'jpg') ;
        close(h)
        ii = ii + 1    
    end

    clear bestVisTime
    clear bestTactTime
    clear vis
    clear tact
    clear bestVisMethod
    clear bestTactMethod
    ind = 1
    for i = 1:2:length(subject),
        indVis = (i-1) * num_ival;
        indTact = i * num_ival;
        bestVisTime(ind,:) = min(loss(indVis+ 1: indVis + num_ival,4:6)');
        bestTactTime(ind,:) = min(loss(indTact+ 1: indTact + num_ival,4:6)');
        bestVisMethod(ind,:) = min(loss(indVis+ 1: indVis + num_ival,4:6));
        bestTactMethod(ind,:) = min(loss(indTact+ 1: indTact + num_ival,4:6));
        %vis(ind:ind+6,1:3) = 
        loss(indVis+ 1: indVis + 6,4:6)
        %tact(ind:ind+6,:) = loss(indTact+ 1: indTact + 6,4:6);
    %     set(gca,'xticklabel',xLab)
    %     legend({'LDA';'Fisher Disc';'LDA Shrink'})
    %     title(fileIndex.files(i,1))
    %     ylim([0 0.5])
        ind = ind + 1;
    end



    %close all
    h=figure()
    subplot(221)
    plot(bestVisTime','o-')
    hold all
    plot(mean(bestVisTime), 'x-')
    set(gca,'xticklabel',xLab)
    NumTicks = num_ival;
    L = get(gca,'XLim');
    set(gca,'XTick',linspace(L(1),L(2),NumTicks)) 
    ylim([0 0.5])
    ylabel('Error %')
    xlabel('Classification Time Window (ms after stimulus)')
    title('Visual Classification Performance by Time')
    hold off

    subplot(222)
    plot(bestTactTime','o-')
    hold all
    plot(mean(bestTactTime), 'x-')
    set(gca,'xticklabel',xLab)
    NumTicks = num_ival;
    L = get(gca,'XLim');
    set(gca,'XTick',linspace(L(1),L(2),NumTicks)) 
    ylim([0 0.5])
    ylabel('Error %')
    xlabel('Time Window (ms after stimulus)')
    title('Tactile Classification Performance by Time')
    hold off


    subplot(223)
    plot(bestVisMethod','o-')
    hold all
    plot(mean(bestVisMethod), 'x-')
    set(gca,'xticklabel',methods)
    NumTicks = 3;
    L = get(gca,'XLim');
    set(gca,'XTick',linspace(L(1),L(2),NumTicks)) 

    ylim([0 0.5])
    ylabel('Error %')
    xlabel('Classifcation Method')
    title('Visual Classification Performance by Method')
    hold off

    subplot(224)
    plot(bestTactMethod','o-')
    hold all
    plot(mean(bestTactMethod), 'x-')
    set(gca,'xticklabel',methods)
    NumTicks = 3;
    L = get(gca,'XLim');
    set(gca,'XTick',linspace(L(1),L(2),NumTicks)) 
    ylim([0 0.5])
    ylabel('Error %')
    xlabel('Classifcation Method')
    title('Tactile Classification Performance by Method')
    
    hold off

    savename=fullfile(LATEX_DIR, [ 'classification_error_' class_results]);
    set(gcf,'paperunits','centimeters')
    set(gcf,'papersize',[22,20])
    saveas(h,savename,'jpg') ;
    close(h)
    
    h=figure()
    subplot(211)
    hold on
    errorbar(mean(bestTactMethod), std(bestTactMethod),'r')
    errorbar(mean(bestVisMethod), std(bestVisMethod),'g')
    set(gca,'xticklabel',methods)
    NumTicks = 3;
    L = get(gca,'XLim');
    set(gca,'XTick',linspace(L(1),L(2),NumTicks)) 
    legend({'Tactile';'Visual'}, 'best')
    ylim([0 0.5])
    ylabel('Error %')
    xlabel('Classifcation Method')
    title('Average Classification Performance by Method')
    hold off
    
    subplot(212)
    
    hold on
    errorbar(mean(bestTactTime), std(bestTactTime),'r')
    errorbar(mean(bestVisTime), std(bestVisTime),'g')
    legend({'Tactile';'Visual'}, 'best')
    NumTicks = num_ival;
    L = get(gca,'XLim');
    set(gca,'XTick',linspace(L(1),L(2),NumTicks)) 
    ylim([0 0.5])
    ylabel('Error %')
    xlabel('Time Window (ms after stimulus)')
    title('Average Classification Performance by Time')
    hold off
    
    savename=fullfile(LATEX_DIR, [ 'avg_classification_error_' class_results]);
    set(gcf,'paperunits','centimeters')
    set(gcf,'papersize',[22,20])
    saveas(h,savename,'jpg') ;
end