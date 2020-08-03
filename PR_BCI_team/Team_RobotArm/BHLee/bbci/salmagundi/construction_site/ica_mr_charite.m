file= 'charite/claudia111002_epi16_seg1';
file_mrk= [EEG_RAW_DIR 'charite/claudia111002_epi16_MRI Artifact Correction.Markers'];

%% read all channels in original sampling rate
cnt= readGenericEEG(file, [], 'raw');

%% read artifact markers
[d,d,pos,d,d]= textread(file_mrk, '%s%s%d%d%s', ...
    'delimiter',',', 'headerlines',2);
mrk.fs= cnt.fs;
mrk.pos= pos';
mrk.toe= ones(size(mrk.pos));

%% remove non-EEG channels
cnt= proc_selectChannels(cnt, 'not','E*');
mnt= setElectrodeMontage(cnt.clab);

%% reduced size of data set to save memory
iv= 1:100000;
X= cnt.x(iv,:)';
clear cnt

tic; W= tdsep0(X, 0:5:100); toc
clear X

%% show scalp patterns of ICs
A= inv(W);
nICs= size(W,1);
for ic= 1:nICs,
  suplot(nICs, ic, .01, .01);
  showScalpPattern(mnt, A(:,ic), 0, 'none', 'range');
  ylabel(sprintf('ic %d', ic));
end


pause


cnt= readGenericEEG(file, [], 'raw');
cnt= proc_selectChannels(cnt, 'not','E*');

%% apply unmixing matrix to data
cnt_ic= copyStruct(cnt, 'x','clab');
cnt_ic.x= cnt.x * W';
for ci= 1:length(cnt.clab),
  cnt_ic.clab{ci}= sprintf('ic%0d', ci);
end

%% show ERP of ICs
epo_ic= makeSegments(cnt_ic, mk, [-50 500]);
epo_ic= proc_baseline(epo_ic, [-50 0]);
mnt_ic= setElectrodeMontage(cnt_ic.clab); 
showERPgrid(epo_ic, mnt_ic, '-');



pause


%% get first 50 segments
mrk_idc= 1:50;
mk= mrk;
mk.pos= mrk.pos(mrk_idc);
epo= makeSegments(cnt, mk, [-500 2000]);
epo= proc_baseline(epo, [-500 50]);

showERPgrid(epo, mnt);


pause


erp= proc_classMean(epo);
iv= getIvalIndices([0 2], erp);
[ma,im]= max(abs(erp.x(iv,:)));
ii= round(mean(im));

%mr_w= mean(erp.x(iv,:))';
mr_w= erp.x(iv(1)+ii-1,:)';

%% plot artifact pattern
clf;
showScalpPattern(mnt, mr_w)

%% projection orthogonal to artifact component
P_demr= eye(length(mr_w)) - mr_w*mr_w'/(mr_w'*mr_w);

%% apply projection to data
epo_demr= proc_linearDerivation(epo, P_demr);
for ci= 1:length(epo.clab),
  epo_demr.clab{ci}= sprintf('comp%0d', ci);
end
mnt_demr= setElectrodeMontage(epo_demr.clab);

%% plot 'corrected' component (#cn)
cn= 1;
clf;
showERP(epo_demr, mnt_demr, cn);

