file= 'bci_competition_ii/albany_P300_train';
xTrials= [10 10];

[cnt, mrk, mnt]= loadProcessedEEG(file);
%cnt= proc_selectChannels(cnt, 'F3-4','FC3-4','C5-6','CP3-4','P7,8','AFz');


lett_matrix= reshape(['A':'Z', '1':'9', ' '], 6, 6)'; 
nTrials= length(mrk.code);
nHighPerLett= 15*12;
nLetts= nTrials/nHighPerLett;
lett= char(zeros(1, nLetts));
target= zeros(nLetts,2);
for il= 1:nLetts,
  iv= [1:12] + (il-1)*nHighPerLett;
  ir= find(mrk.toe(iv)==1);
  target(il,:)= sort(mrk.code(iv(ir))) - [0 6];
  lett(il)= lett_matrix(target(il,2), target(il,1));
end
lett


epo= makeEpochs(cnt, mrk, [0 410]);
fv= proc_selectChannels(epo, 'F3-4','FC3-4','C5-6','CP3-4', ...
                             'P7,8','AFz');
fv= proc_baseline(fv, [0 150]);
fv= proc_selectIval(fv, [200 401]);
fv= proc_jumpingMeans(fv, 10);

[em,es,out,avErr,evErr]= doXvalidationPlus(fv, 'LDA', xTrials, 3);
%% 22.4±0.2%  (fn: 26.9±0.4%,  fp: 21.7±0.2%)  [train: 21.4±0.0%]

idx_dev= find(mrk.y(1,:));
d= [0 diff(idx_dev)];

d_rng= 1:max(d);
for id= d_rng,
  idx= find(d==id);
  avd(id)= mean(evErr(idx_dev(idx)));
end
plot(d_rng, avd(d_rng));


idx_sel= find(d>2);
mrk_sel= mrk_selectEvents(mrk, [idx_dev(idx_sel), ...
                                find(mrk.y(2,:))]);
epo= makeEpochs(cnt, mrk_sel, [0 410]);
fv= proc_selectChannels(epo, 'F3-4','FC3-4','C5-6','CP3-4', ...
                             'P7,8','AFz');
fv= proc_baseline(fv, [0 150]);
fv= proc_selectIval(fv, [200 401]);
fv= proc_jumpingMeans(fv, 10);

[em,es,out,avErr,evErr]= doXvalidationPlus(fv, 'LDA', xTrials, 3);
%% 19.6±0.1%  (fn: 21.6±0.4%,  fp: 19.0±0.2%)  [train: 18.5±0.0%]

d_sel= d(idx_sel);
avd_sel= NaN*ones(size(avd));
d_rng= min(d_sel):max(d_sel);
for id= d_rng,
  idx= find(d_sel==id);
  avd_sel(id)= mean(evErr(idx));
end
plot(1:max(d), [avd' avd_sel']);



mrk_dev= mrk_setClasses(mrk, ...
                        {idx_dev(find(d==1)); ...
                         idx_dev(find(d==2)); ...
                         idx_dev(find(d==3)); ...
                         idx_dev(find(d==4)); ...
                         idx_dev(find(d==5)); ...
                         idx_dev(find(d==6))}, ...
                        {'deviant1', 'deviant2', 'deviant3', ...
                         'deviant4', 'deviant5', 'deviant6'});
mrk_dev= mrk_selectEvents(mrk_dev, any(mrk_dev.y));

grd= sprintf('F7,F3,Fz,F4,F8\nT7,C3,Cz,C4,T8\nP7,CP3,CPz,CP4,P8\nPO7,O1,legend,O2,PO8');
mnt_large= setDisplayMontage(mnt, grd);
opt= struct('colorOrder','rainbow');
epo= makeEpochs(cnt, mrk_dev, [-50 500]);
epo= proc_baseline(epo, [-50 0]);
epo= proc_movingAverage(epo, 25);
grid_plot(epo, mnt_large, opt);
