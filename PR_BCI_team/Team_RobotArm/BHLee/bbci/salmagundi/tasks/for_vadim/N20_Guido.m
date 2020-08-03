file= strcat('Guido_02_04_12/', {'medianus_left', 'medianus_right', ...
				 'tibialis_right'}, 'Guido');
className= {'left medianus', 'right medianus', 'right tibialis'};

grid_opt= struct('axisTitleFontWeight','bold', ...
                 'colorOrder',[1 0 0; 0 0.7 0; 0 0 1]);

for gg= 1:length(file),
  Cnt= readGenericEEG(file{gg}, [], 'raw');
  Mrk= readMarkerTable(file{gg}, Cnt.fs);
  Mrk= makeClassMarkers(Mrk, {32});
  Mrk.className= className(gg);
  if gg==1,
    cnt= Cnt;
    mrk= Mrk;
  else
    [cnt, mrk]= proc_appendCnt(cnt,Cnt, mrk,Mrk);
    clear Cnt Mrk;
  end
end
mnt= getElectrodePositions(cnt.clab);
mnt= mnt_setGrid(mnt, 'motor_cortex_w7');

%cnt= proc_commonAverageReference(cnt);
%cnt= proc_laplace(cnt, 'vertical');

epo= makeEpochs(cnt, mrk, [-50 100]);
epo= proc_baseline(epo, [-50 0]);
%epo= proc_baseline(epo, [0 15]);
%epo_rsq= proc_r_square(proc_selectClasses(epo, 1:2));

grid_plot(epo, mnt, grid_opt);
%grid_addBars(epo_rsq, 'box','on');


scalpEvolutionPlusChannel(epo, mnt, {'C3','Cz'}, ...
                          [33 40; 43 50; 53 60; 85 95], ...
                          'colorOrder',grid_opt.colorOrder, 'legendPos',2);

scalpEvolutionPlusChannel(epo, mnt, {'CP3','FCz'}, ...
			  [23 30; 33 40; 43 50; 53 60; 85 95], ...
                          'colorOrder',grid_opt.colorOrder, 'legendPos',2);


epo= proc_baseline(epo, [15 19]);
grid_plot(epo, mnt, grid_opt);

scalpEvolutionPlusChannel(epo, mnt, {'CP3','FCz'}, ...
			  [24 27; 29 32; 36 39; 57 60], ...
			  'xTick',[-50 0:10:50 100], ...
                          'colorOrder',grid_opt.colorOrder, 'legendPos',2);
printFigure('temp/guido_N20', [20 16]);

