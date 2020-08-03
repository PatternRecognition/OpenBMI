file= 'Guido_05_04_01/imag_lettGuido';
classDef= {1, 2; 'left','foot'};
%file= strcat('Guido_04_03_29/imag_', {'lett','move'}, 'Guido');
%classDef= {1, 3; 'left','foot'};

csp.clab= {'not','E*','Fp*','AF*','I*','OI*',...
           'OPO*','TP9,10','T9,10','FT9,10'};
csp.band= [7 13];
csp.filtOrder= 5;
csp.ival= [750 3500];
csp.nPat= 3;

cnt= readGenericEEG(file);
cnt= proc_selectChannels(cnt, 'not','x*');  %% only for two player exp
cnt= proc_selectChannels(cnt, csp.clab);
[b,a]= butter(csp.filtOrder, csp.band/cnt.fs*2);
cnt= proc_filt(cnt, b, a);
mrk= readMarkerTable(file);
mrk= makeClassMarkers(mrk, classDef);
mnt= getElectrodePositions(cnt.clab);


%%%
file_arte= '/cdwriter/Guido_04_03_29/arteGuido';
%file_arte= 'Guido_04_03_29/arteGuido';
ct= readGenericEEG(file_arte);
mk= readMarkerComments(file_arte);
pos_op= strmatch('Augen offen', mk.str);
pos_cl= strmatch('Augen zu', mk.str);
mrk_arte= struct('pos', mk.pos([pos_op pos_cl]), 'fs',mk.fs);
mrk_arte.y= [1 0; 0 1];
mrk_arte.className= {'eyes open', 'eyes close'};
len= min(diff(mk.pos(pos_op+[0 1])), diff(mk.pos(pos_cl+[0 1])));
len_ms= len/mk.fs*1000;

ct= proc_selectChannels(ct, cnt.clab);  %% only when 64ch exp
ct= proc_selectChannels(ct, csp.clab);
ct= proc_filt(ct, b, a);
%epo= makeEpochs(ct, mrk_arte, [0 len_ms]);
%[fv, csp_w, csp_la, csp_a]= proc_csp2(epo, 1);
%plotCSPanalysis(epo,mnt,csp_w,csp_a,csp_la)

blk= struct('fs',mk.fs, 'ival',mk.pos([pos_cl pos_cl+1])'+[5; 0]*mrk.fs);
mrk_ha= mrk_evenlyInBlocks(blk, diff(csp.ival));
epo_ha= makeEpochs(ct, mrk_ha, [0 diff(csp.ival)]);
blk= struct('fs',mk.fs, 'ival',mk.pos([pos_op pos_op+1])'+[5; 0]*mrk.fs);
mrk_la= mrk_evenlyInBlocks(blk, diff(csp.ival));
epo_la= makeEpochs(ct, mrk_la, [0 diff(csp.ival)]);
Nha= length(mrk_ha.pos);
Nha2= floor(Nha/2);
%%%


epo= makeEpochs(cnt, mrk, csp.ival);

[fv, csp_w, csp_la, csp_a]= proc_csp2(epo, csp.nPat);
figure(1); clf;
plotCSPanalysis(epo,mnt,csp_w,csp_a,csp_la)
fv= proc_variance(fv);
fv= proc_logarithm(fv);

nTrials= size(fv.x,3);
iTr= [1:ceil(nTrials/2)];
iTe= [ceil(nTrials/2)+1:nTrials];
cl1= find(fv.y(1,iTe));
cl2= find(fv.y(2,iTe));

C= trainClassifier(fv, 'LDA', iTr);
out= applyClassifier(fv, 'LDA', C, iTe);
err= 100*mean(sign(out)~=[-1 1]*fv.y(:,iTe));

figure(2); clf;
plot(cl1, out(cl1), 'r.'); hold on
plot(cl2, out(cl2), 'g.'); 

add_idx= mod([1:size(epo.x,3)]-1, Nha2)+1;
epo_no= epo;
epo_no.x= [epo.x + epo_ha.x(:,:,add_idx)] / 2;

fv= proc_linearDerivation(epo_no, csp_w);
fv= proc_variance(fv);
fv= proc_logarithm(fv);
out_no= applyClassifier(fv, 'LDA', C, iTe);
err_no= 100*mean(sign(out_no)~=[-1 1]*fv.y(:,iTe));

plot(cl1, out_no(cl1), 'rx'); hold on
plot(cl2, out_no(cl2), 'gx'); hold off
title(sprintf('error: %.1f -> %.1f', err, err_no));

epo_app= proc_selectEpochs(epo_ha, Nha2+1:Nha);
epo_app.y= ones(1,size(epo_app.x,3));
epo_app.className= {'left'};
epo2= proc_appendEpochs(epo, epo_app);
epo_app.className= {'foot'};
epo2= proc_appendEpochs(epo2, epo_app);
epo_app= epo_la;
epo_app.y= ones(1,size(epo_app.x,3));
epo_app.className= {'left'};
epo2= proc_appendEpochs(epo, epo_app);
epo_app.className= {'foot'};
epo2= proc_appendEpochs(epo2, epo_app);

[fv, csp_w, csp_la, csp_a]= proc_csp2(epo2, csp.nPat);
figure(3); clf;
plotCSPanalysis(epo2,mnt,csp_w,csp_a,csp_la)
%fv= proc_variance(fv);
%fv= proc_logarithm(fv);
fv= proc_linearDerivation(epo_no, csp_w);
fv= proc_variance(fv);
fv= proc_logarithm(fv);
out_no= applyClassifier(fv, 'LDA', C, iTe);
err_no= 100*mean(sign(out_no)~=[-1 1]*fv.y(:,iTe));

figure(4); clf;
plot(cl1, out(cl1), 'r.'); hold on
plot(cl2, out(cl2), 'g.'); 
plot(cl1, out_no(cl1), 'rx'); hold on
plot(cl2, out_no(cl2), 'gx'); hold off
title(sprintf('error: %.1f -> %.1f', err, err_no));
