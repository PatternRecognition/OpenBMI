my_list= {'AA', [11 14; 23 27], [1000 3000], ...
          'BB', [12 15; 25 28], [1000 3000], ...
          'CC', [10 14; 23 26], [1000 3000]};
csp_info= construct(my_list, 'subject', 'band', 'ival');

sub_dir= 'bci_competition_ii/';
cd([BCI_DIR 'tasks/' sub_dir]);

for is= 1:length(csp_info),
subject= csp_info(is).subject;
dscr_band= csp_info(is).band;
csp_ival= csp_info(is).ival;
  
file= sprintf('%salbany_%s_train', sub_dir, subject);

clear cnt
[cnt, Mrk, mnt]= loadProcessedEEG(file);
mrk= mrk_selectClasses(Mrk, {'top','bottom'});

band= dscr_band(1,:) + [-1 1];
[b,a]= getButterFixedOrder(band, cnt.fs, 6);
cnt_flt= proc_filt(cnt, b, a);
csp= makeEpochs(cnt_flt, mrk, csp_ival);
clear cnt_flt
[csp, w1, la1]= proc_csp(csp, 2);
[so, si]= sort(-la1);
band1_good= si(1:2);

band= dscr_band(2,:) + [-1 1];
[b,a]= getButterFixedOrder(band, cnt.fs, 6);
cnt_flt= proc_filt(cnt, b, a);
csp= makeEpochs(cnt_flt, mrk, csp_ival);
clear cnt_flt
[csp, w2, la2]= proc_csp(csp, 2);
[so, si]= sort(-la2);
band2_good= si(1:2);

csp_w= [w1(:,band1_good), w2(:,band2_good)];
csp_la= [la1(band1_good), la2(band2_good)];

csp_clab= {'band1:csp1','band1:csp2','band2:csp1','band2:csp2'};

save(['albany_csp_' subject], 'csp_w','csp_la','csp_clab',...
     'dscr_band','csp_ival');

end
