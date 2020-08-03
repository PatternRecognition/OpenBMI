%file= 'Guido_05_04_01/imag_lettGuido';
file= strcat('Guido_04_03_29/imag_', {'lett','move'}, 'Guido');
classes= {'left','foot'};

csp.clab= {'not','E*','Fp*','AF*','I*','OI*',...
           'OPO*','TP9,10','T9,10','FT9,10'};
csp.band= [7 13];
csp.filtOrder= 5;
csp.ival= [750 3500];
csp.nPat= 3;

[cnt, mrk, mnt]= eegfile_loadMatlab(file);
mrk= mrk_selectClasses(mrk, classes);
cnt= proc_selectChannels(cnt, csp.clab);
[b,a]= butter(csp.filtOrder, csp.band/cnt.fs*2);
cnt= proc_filt(cnt, b, a);


%%%
if iscell(file),
  fil= file{1};
else
  fil= file;
end
subdir= fileparts(fil);
ii= max(find(fil==upper(fil)));
sbj= fil(ii:end);
file_arte= ['/cdrom/' subdir '/arte' sbj];
ct= eegfile_loadBV(file_arte, 'fs',100);
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
[bb,aa]= butter(csp.filtOrder, [4 35]/cnt.fs*2);
ct= proc_filt(ct, bb, aa);

%% CSP would do overfitting
%epo= makeEpochs(ct, mrk_arte, [0 len_ms]);
%[fv, csp_w, csp_la, csp_a]= proc_csp3(epo, 1);
%plotCSPanalysis(epo,mnt,csp_w,csp_a,csp_la)

%% use ICA
ica_w= tdsep0(ct.x', 0:25)';
ep= makeEpochs(ct, mrk_arte, [0 len_ms]);
mt= mnt_adaptMontage(mnt, ep); 
%plotPatternsPlusSpec(ep, mt, ica_w');
ic= proc_linearDerivation(ct, ica_w, 'prependix','ica');
ic.clab= apply_cellwise(ic.clab, 'strrep',' ','0');

blk= struct('fs',mk.fs, 'ival',mk.pos([pos_cl pos_cl+1])'+[5; 0]*mrk.fs);
mrk_ha= mrk_evenlyInBlocks(blk, 1000);
mrk_ha.y= ones(1, length(mrk_ha.pos));
mrk_ha.className= {'eyes closed'};
blk= struct('fs',mk.fs, 'ival',mk.pos([pos_op pos_op+1])'+[5; 0]*mrk.fs);
mrk_la= mrk_evenlyInBlocks(blk, 1000);
mrk_la.y= ones(1, length(mrk_la.pos));
mrk_la.className= {'eyes open'};
mrk_oc= mrk_mergeMarkers(mrk_ha, mrk_la);
ep_ic= makeEpochs(ic, mrk_oc, [0 990]);
spec= proc_spectrum(ep_ic, [7 13]);
ep_icr= proc_r_square_signed(spec);
ep_icr= proc_pickAmplitudePeak(ep_icr, [], 'max');
[so,si]= sort(-ep_icr.x);
plotPatternsPlusSpec(ep, mt, ica_w', 'selection',si(1:14));

nComp= 5;
sel= si(1:nComp);
del= setdiff(1:size(ica_w,1), sel);
%ics= proc_selectChannels(ic, sel);
ics= ic;
ics.x(:,del,:)= 0;
ct_alpha= proc_linearDerivation(ics, inv(ica_w));
ct_alpha.clab= ct.clab;

ep_alpha= makeEpochs(ct_alpha, mrk_oc, [0 990]);
spec= proc_spectrum(ep_alpha, [3 35], hamming(ep_alpha.fs));
grid_plot(spec, mnt);

blk= struct('fs',mk.fs, 'ival',mk.pos([pos_cl pos_cl+1])'+[5; 0]*mrk.fs);
mrk_ha= mrk_evenlyInBlocks(blk, diff(csp.ival));
ep_ha= makeEpochs(ct_alpha, mrk_ha, [0 diff(csp.ival)]);
Nha= length(mrk_ha.pos);
Nha2= floor(Nha/2);
%%%



epo= makeEpochs(cnt, mrk, csp.ival);
nTrials= size(epo.x,3);
iTr= [1:ceil(nTrials/2)];
iTe= [ceil(nTrials/2)+1:nTrials];
iTe1= find(epo.y(1,iTe));
iTe2= find(epo.y(2,iTe));

epo_tr= proc_selectEpochs(epo, iTr);
[fv, csp_w, csp_la, csp_a]= proc_csp2(epo_tr, csp.nPat);
%figure(1); clf;
%plotCSPanalysis(epo_tr, mnt, csp_w, csp_a, csp_la)
fv= proc_variance(fv);
fv= proc_logarithm(fv);
C= trainClassifier(fv, 'LDA');
fv= proc_selectEpochs(epo, iTe);
fv= proc_linearDerivation(fv, csp_w);
fv= proc_variance(fv);
fv= proc_logarithm(fv);
out= applyClassifier(fv, 'LDA', C);
err= 100*mean(sign(out)~=[-1 1]*fv.y);

figure(2); clf;
plot(iTe1, out(iTe1), 'r.'); hold on
plot(iTe2, out(iTe2), 'g.'); 



add_idx= mod([1:length(iTe)]-1, Nha2)+1;
epo_no= proc_selectEpochs(epo, iTe);
epo_no.x= [epo_no.x + ep_ha.x(:,:,add_idx)] / 2;

fv= proc_linearDerivation(epo_no, csp_w);
fv= proc_variance(fv);
fv= proc_logarithm(fv);
out_no= applyClassifier(fv, 'LDA', C);
err_no= 100*mean(sign(out_no)~=[-1 1]*fv.y);

plot(iTe1, out_no(iTe1), 'rx'); hold on
plot(iTe2, out_no(iTe2), 'gx'); hold off
title(sprintf('error: %.1f -> %.1f', err, err_no));




%%%
%%% --- go for it
%%%


epo_tr= proc_selectEpochs(epo, iTr);
sx= size(epo_tr.x);
idx= find(epo_tr.y(1,:));
z= permute(epo_tr.x(:,:,idx),[1 3 2]);
z= reshape(z, [sx(1)*length(idx) sx(2)]);
R1= cov(z);
idx= find(epo_tr.y(2,:));
z= permute(epo_tr.x(:,:,idx),[1 3 2]);
z= reshape(z, [sx(1)*length(idx) sx(2)]);
R2= cov(z);
[W,D]= eig(R1, R1+R2);

ww= W(:,1);

fv= proc_linearDerivation(epo_tr, ww);
fv= proc_variance(fv);
fv= proc_logarithm(fv);
C= trainClassifier(fv, 'LDA');
fv= proc_selectEpochs(epo, iTe);
fv= proc_linearDerivation(fv, ww);
fv= proc_variance(fv);
fv= proc_logarithm(fv);
out= applyClassifier(fv, 'LDA', C);
err= 100*mean(sign(out)~=[-1 1]*fv.y);

clf;
plot(iTe1, out(iTe1), 'r.'); hold on
plot(iTe2, out(iTe2), 'g.'); 


sx= size(ep_ha.x);
z= permute(ep_ha.x,[1 3 2]);
z= reshape(z, [sx(1)*sx(3) sx(2)]);
T= cov(z);
T= T/trace(T);
R1= R1/trace(R1);
R2= R2/trace(R2);
[W,D]= eig(R1, R1+R2+10*T);
ww= W(:,1);

fv= proc_linearDerivation(epo_tr, ww);
fv= proc_variance(fv);
fv= proc_logarithm(fv);
C= trainClassifier(fv, 'LDA');
fv= proc_selectEpochs(epo, iTe);
fv= proc_linearDerivation(fv, ww);
fv= proc_variance(fv);
fv= proc_logarithm(fv);
out= applyClassifier(fv, 'LDA', C);
err= 100*mean(sign(out)~=[-1 1]*fv.y)

clf;
plot(iTe1, out(iTe1), 'r.'); hold on
plot(iTe2, out(iTe2), 'g.'); 

fv= proc_linearDerivation(epo_no, ww);
fv= proc_variance(fv);
fv= proc_logarithm(fv);
out_no= applyClassifier(fv, 'LDA', C);
err_no= 100*mean(sign(out_no)~=[-1 1]*fv.y);

plot(iTe1, out_no(iTe1), 'rx'); hold on
plot(iTe2, out_no(iTe2), 'gx'); hold off
title(sprintf('error: %.1f -> %.1f', err, err_no));




