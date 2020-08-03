EEG_MAT_DIR= 'C:\uni\lab rotation\bbciMat\';
LATEX_DIR= 'C:\uni\lab rotation\images\';
close all
clear subject Cz_r2
count = 0
do_images = 1
if count == 1,
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
% subject(1).name='VPnh_vis_count';
% subject(1).path='Thomas_09_10_23/Thomas_vis_count'; 
% subject(2).name='VPnh_tact_count';
% subject(2).path='Thomas_09_10_23/Thomas_tact_count';
% subject(3).name='VPmk_vis_count';
% subject(3).path='sophie_09_10_30/sophie_vis_count';
% subject(4).name='VPmk_tact_count';
% subject(4).path='sophie_09_10_30/sophie_tact_count';
% subject(5).name='VPgao_vis_count';
% subject(5).path='chris_09_11_17/chris_vis_count';
% subject(6).name='VPgao_tact_count';
% subject(6).path='chris_09_11_17/chris_tact_count';
subject(1).name='VPiac_vis_count';
subject(1).path='nico_09_11_12/nico_vis_count';
subject(2).name='VPiac_tact_count';
subject(2).path='nico_09_11_12/nico_tact_count';
% subject(9).name='rithwick_vis_count';
% subject(9).path='rithwick_09_11_05/rithwick_vis_count';   
% subject(10).name='rithwick_tact_count';
% subject(10).path='rithwick_09_11_05/rithwick_tact_count';
end
base_ival = [-100 0];
epo_ival = [-100 700];
disp_ival =[-50 700];
crit_ival= [100 700];

clab= {'FCz', 'Pz', 'CPz'};
clab2= {'FCz', 'Pz'};
opt_scalp= {'resolution', 30, 'extrapolate', 1};

for sub=1:length(subject),

    clear cnt mrk mnt epo epo_r
    [cnt, mrk, mnt]= eegfile_loadMatlab(subject(sub).path);

    [mk, rClab]= reject_varEventsAndChannels(cnt, mrk, disp_ival, ...
                                              'visualize', 0);
    mrk= mk;

    epo= cntToEpo(cnt, mrk,epo_ival);
    epo= proc_baseline(epo, base_ival);
    Pz_ind = strmatch('Pz',epo.clab);
    %epo= proc_detrend(epo);
    epo_crit= proc_selectIval(epo, crit_ival);
    iArte= find_artifacts(epo_crit, 'EOG*', struct('maxmin', 150));
    epo= proc_selectEpochs(epo, 'not',iArte);
    epo= proc_selectIval(epo, disp_ival);
    epo_r= proc_r_square_signed(epo);
    Cz_r2(sub,:) = epo_r.x(:,Pz_ind)
    
        

    if do_images == 1,

        [ival_scalps, nfo]= select_time_intervals(epo_r, 'visualize', 0, ...
                           'visu_scalps', 0, 'clab',{'not','E*','Fp*','AF*'});
        ival_scalps= visutil_correctIvalsForDisplay(ival_scalps, 'fs',epo.fs);
        h = figure();
        H= grid_plot(epo, mnt, defopt_erps,'scalePolicy',[-10,15]);
        grid_addBars(epo_r, 'h_scale',H.scale);

        savename=fullfile(LATEX_DIR, [subject(sub).name, '_grid']);
        set(gcf,'paperunits','centimeters')
        set(gcf,'papersize',[22,20])
        %set(gcf,'paperposition',[0.5,0.5,11,2])
        saveas(h,savename,'jpg') ;
        close all;
        disp('pics1 saved');




        h = figure();

        scalpEvolutionPlusChannel(epo_r, mnt, clab, ival_scalps, ...
                                     defopt_scalp_power, opt_scalp{:}, 'legend_pos','NorthWest','yLim',[-0.01 0.06]);
        savename=fullfile(LATEX_DIR, [subject(sub).name, '_scalp_r2']);
        set(gcf,'paperunits','centimeters')
        set(gcf,'papersize',[22,20])
        %set(gcf,'paperposition',[0.5,0.5,11,2])
        saveas(h,savename,'jpg') ;
        close all;
        disp('pics2 saved');


        h = figure();

        scalpEvolutionPlusChannel(epo, mnt, clab2, ival_scalps, ...
                                     defopt_scalp_power, opt_scalp{:}, 'legend_pos','NorthWest', 'yLim',[-8 8]);
        savename=fullfile(LATEX_DIR, [subject(sub).name, '_scalp_v']);
        set(gcf,'paperunits','centimeters')
        set(gcf,'papersize',[22,20])
        %set(gcf,'paperposition',[0.5,0.5,11,2])
        saveas(h,savename,'jpg') ;
        close all;
        disp('pics3 saved');
    end;    
end;

Cz_r2_vis = Cz_r2([1,3,5,7,9],:)
Cz_r2_tact = Cz_r2([2,4,6,8,10],:)