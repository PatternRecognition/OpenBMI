file= 'bci_competition_ii/albany_P300_train';
fig_dir= 'bci_competition_ii/';

%[Epo, mrk, mnt]= loadProcessedEEG(file, 'avg_15');
[cnt, mrk, mnt]= loadProcessedEEG(file);

Epo= makeEpochs(cnt, mrk, [-50 550]);
clear cnt


mnt.x(64)= 0;
mnt.y(64)= [-0.6 1.6]*mnt.y([58 62]);
mnt.x(44)= [-0.6 1.6]*mnt.x([14 42]);
mnt.y(44)= 0;
mnt.x(43)= [-0.6 1.6]*mnt.x([8 41]);
mnt.y(43)= 0;
grd= sprintf('T7,P7,legend,P8,T8\nP3,P1,Pz,P2,P4\nP5,PO3,POz,PO4,P6\nPO7,O1,Oz,O2,PO8');
mnt_visual= setDisplayMontage(mnt, grd);


epo= proc_albanyAverageP300Trials(Epo, 15);
iRow= find(epo.code>6);
epo= proc_selectEpochs(epo, iRow);
epo.code= epo.code(iRow);
fv= proc_selectChannels(epo, 'T8,10','CP7,8','P#','PO#','O#','Iz');
fv= proc_baseline(fv, [0 150]);
fv= proc_selectIval(fv, [220 450]);
fv= proc_jumpingMeans(fv, 10);

model= struct('classy','RLDA', 'msDepth',3, 'inflvar',1);
model.param= [0 0.01 0.1 0.5];
classy= selectModel(fv, model, [3 10 round(9/10*sum(any(fv.y)))]);
[em,es,out,avErr,evErr]= ...
    doXvalidationPlus(fv, classy, [100 10], 3);
outliers= find(evErr>0.05)

epo.className= {'correct row', 'wrong row'};
erp= proc_average(epo);
erp= proc_baseline(erp, [0 150]);

fac= 0.4;
for io= 1:length(outliers),
  erp.title= sprintf('albany P300, %g*outlier #%d (%d%%)', ...
                     fac, io, round(100*evErr(outliers(io))));
  grid_plot(erp, mnt_visual);

  out= proc_selectEpochs(epo, outliers(io));
  out.x= fac*out.x;
  out= proc_baseline(out, [0 150]);
  out.className{1}= [out.className{1} ' outlier'];
  grid_add_plot(out, mnt_visual);
  grid_markIval([220 450]);

  saveFigure([fig_dir 'albany_P300_row_outlier' int2str(io)], [10 6]*2);
  
  oo= outliers(io);
  iTrial= floor((oo-1)/6);
  iSubTrials= iTrial*12*15+[1:12*15];
  iCodeMatch= find(Epo.code(iSubTrials)==epo.code(oo));
  outSubEpochs= iSubTrials(iCodeMatch);
  out= proc_selectEpochs(Epo, outSubEpochs);
  out= proc_baseline(out, [0 150]);
  out.className= {[out.className{1} ' row']};
  out.y= eye(15);
  opt.colorOrder= 'rainbow';
  grid_plot(out, mnt_visual, opt);

  saveFigure([fig_dir 'albany_P300_row_subtrials_outlier' int2str(io)]);
end



epo= proc_albanyAverageP300Trials(Epo, 15);
iCol= find(epo.code<=6);
epo= proc_selectEpochs(epo, iCol);
epo.code= epo.code(iCol);
fv= proc_selectChannels(epo, 'P7-3','P4-8','PO#','O#','Iz');
fv= proc_baseline(fv, [150 200]);
fv= proc_selectIval(fv, [220 450]);
fv= proc_jumpingMeans(fv, 10);
model= struct('classy','RLDA', 'msDepth',3, 'inflvar',1);
model.param= [0 0.01 0.1 0.5];
classy= selectModel(fv, model, [3 10 round(9/10*sum(any(fv.y)))]);
[em,es,out,avErr,evErr]= ...
    doXvalidationPlus(fv, classy, [100 10], 3);
outliers= find(evErr>0.05)

epo.className= {'correct col', 'wrong col'};
erp= proc_average(epo);
erp= proc_baseline(erp, [150 200]);

fac= 0.4;
for io= 1:length(outliers),
  erp.title= sprintf('albany P300, %g*outlier #%d (%d%%)', ...
                     fac, io, round(100*evErr(outliers(io))));
  grid_plot(erp, mnt_visual);

  out= proc_selectEpochs(epo, outliers(io));
  out.x= fac*out.x;
  out= proc_baseline(out, [150 200]);
  out.className{1}= [out.className{1} ' outlier'];
  grid_add_plot(out, mnt_visual);
  grid_markIval([220 450]);

  saveFigure([fig_dir 'albany_P300_col_outlier' int2str(io)], [10 6]*2);
  
  oo= outliers(io);
  iTrial= floor((oo-1)/6);
  iSubTrials= iTrial*12*15+[1:12*15];
  iCodeMatch= find(Epo.code(iSubTrials)==epo.code(oo));
  outSubEpochs= iSubTrials(iCodeMatch);
  out= proc_selectEpochs(Epo, outSubEpochs);
  out= proc_baseline(out, [0 150]);
  out.className= {[out.className{1} ' col']};
  out.y= eye(15);
  opt.colorOrder= 'rainbow';
  grid_plot(out, mnt_visual, opt);

  saveFigure([fig_dir 'albany_P300_row_subtrials_outlier' int2str(io)]);
end
