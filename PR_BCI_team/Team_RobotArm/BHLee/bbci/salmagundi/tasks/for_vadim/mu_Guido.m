file_list= {strcat('Guido_02_01_08/medianus_', {'left','right'}, 'Guido'),
            strcat('Guido_02_04_12/', {'medianus_left', 'medianus_right', ...
                    'tibialis_right'}, 'Guido')};
className= {'left medianus', 'right medianus', 'right tibialis'};

grid_opt= struct('axisTitleFontWeight','bold', ...
                 'colorOrder',[1 0 0; 0 0.7 0; 0 0 1]);
spec_opt= grid_opt;
spec_opt.xTick= 10:10:40;

band= [9 11.5];


ff= 2;
%for ff= 1:length(file_list),

file= file_list{ff};
for gg= 1:length(file),
  Cnt= readGenericEEG(file{gg});
  Mrk= readMarkerTable(file{gg});
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
mnt= mnt_excludeFromGrid(mnt, 'E*');

cnt= proc_commonAverageReference(cnt);
%cnt= proc_laplace(cnt, 'vertical');

epo= makeEpochs(cnt, mrk, [-500 500]);
spec= proc_spectrum(epo, [5 45]);
spec_rsq= proc_r_square(proc_selectClasses(spec, 1:2));

grid_plot(spec, mnt, spec_opt);
grid_markIval(band);
grid_addBars(spec_rsq, 'box','on');

scalpPatterns(spec, mnt, band, 'colorOrder',grid_opt.colorOrder);


[b,a]= butter(2, band/cnt.fs*2);
cnt_flt= proc_filtfilt(cnt, b, a);
erd= makeEpochs(cnt_flt, mrk, [-500 500]);
erd= proc_rectifyChannels(erd);
erd= proc_baseline(erd, [-500 -350]);
erd_rsq= proc_r_square(proc_selectClasses(erd, 1:2));
erd= proc_average(erd);
erd= proc_movingAverage(erd, 200, 'centered');
erd= proc_baseline(erd, [-500 -350]);

grid_plot(erd, mnt, grid_opt);
grid_addBars(erd_rsq);

scalpEvolutionPlusChannel(erd, mnt, {'C3','C4'}, ...
                          [-200 -100; -100 0; 50 150; 150 250; 250 350], ...
                          'colorOrder',grid_opt.colorOrder);

%end
