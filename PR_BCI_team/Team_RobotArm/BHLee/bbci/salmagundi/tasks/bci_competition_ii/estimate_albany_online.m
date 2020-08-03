data_dir= [DATA_DIR 'eegImport/bci_competition_ii/albany/'];

cuts=[ -12 1 15; -18 -4 13; -13 4 20];

clab= {'FC5','FC3','FC1','FCz','FC2','FC4','FC6', ...
       'C5','C3','C1','Cz','C2','C4','C6', ...
       'CP5','CP3','CP1','CPz','CP2','CP4','CP6', ...
       'Fp1','Fpz','Fp2', 'AF7','AF3','AFz','AF4','AF8', ...
       'F7','F5','F3','F1','Fz','F2','F4','F6','F8', ...
       'FT7','FT8','T7','T8','T9','T10','TP7','TP8', ...
       'P7','P5','P3','P1','Pz','P2','P4','P6','P8', ...
       'PO7','PO3','POz','PO4','PO8','O1','Oz','O2','Iz'};
testset_runs= 7:10;

sub_dir= 'bci_competition_ii/';
cd([BCI_DIR 'tasks/' sub_dir]);

%classy= 'LDA';
%model= struct('classy','RLDA', 'msDepth',2, 'inflvar',1);
%model.param= [0 0.01 0.1 0.5 0.8];

Subject= {'AA','BB','CC'},
 
subject= Subject{1};

for Subject={'AA','BB','CC'},

subject=Subject{1};

load(['albany_csp_' subject]);
fprintf('%s:  band1= [%d %d],  band2= [%d %d],  csp_ival= [%d %d]\n', ...
        subject, dscr_band(1,:), dscr_band(2,:), csp_ival);
file= sprintf('%salbany_%s_train', sub_dir, subject);

[cnt, Mrk, mnt]= loadProcessedEEG(file);
mrk= mrk_selectClasses(Mrk, {'top','bottom'});

dat= proc_linearDerivation(cnt, csp_w);
dat.clab= csp_clab;

epotr=makeEpochs(dat, mrk, [0 4000]);

intervals=[750:250:2500; 1250:250:3000]';

clear feat1 feat2 prov feat;

for i=1:size(intervals,1)                        %collect intervals' Fourier coefficients
                                                 % as separate training examples
  prov=proc_selectIval(epotr,intervals(i,:));
  feat1=proc_fourierBandMagnitude(prov,dscr_band(1,:));
  feat2=proc_fourierBandMagnitude(prov,dscr_band(2,:));
  if (i==1)
    feat=proc_catFeatures(feat1,feat2);
  else                                           % merges intervals as separate examples
    prov=proc_catFeatures(feat1,feat2);
    feat.x=cat(3,feat.x,prov.x);
    feat.y=cat(2,feat.y,prov.y);
  end;
end;

clear feat1 feat2 prov;

model= struct('classy','RLDA', 'msDepth',2, 'inflvar',1);
model.param= [0 0.01 0.1 0.25 0.5 0.75 1];

classy= selectModel(feat, model, [10 10]);
C= trainClassifier(feat, classy);

epotr=makeEpochs(dat, Mrk, [0 4000]);        % now consider all classes (to find cut values)

intervals=[750:250:2500; 1250:250:3000]';

for i=1:size(intervals,1)
  prov=proc_selectIval(epotr,intervals(i,:));
  feat1=proc_fourierBandMagnitude(prov,dscr_band(1,:));
  feat2=proc_fourierBandMagnitude(prov,dscr_band(2,:));
  feat(i)=proc_catFeatures(feat1,feat2);
end;

clear feat1 feat2 prov;
clear zr zt z res;

for i=1:size(intervals,1); zr(i,:)=applyClassifier(feat(i),model,C); end;


th=3;
zt=zr;
zt(zr<-th)=-th;
zt(zr>th)=th;

z=sum(zt);                     %Now find best cut values between classes on training set

val=0:-1:-30;
for i=1:length(val)
res(i)=length(find(z(epotr.y(1,:)>0)>=val(i)))+length(find(z(epotr.y(2,:)>0)<val(i)));
end;

[dummy,index]=min(res);
cut(1)=val(index);

val=-15:1:15;
for i=1:length(val)
res(i)=length(find(z(epotr.y(2,:)>0)>=val(i)))+length(find(z(epotr.y(3,:)>0)<val(i)));
end;

[dummy,index]=min(res);
cut(2)=val(index);

val=0:1:30;
for i=1:length(val)
res(i)=length(find(z(epotr.y(3,:)>0)>=val(i)))+length(find(z(epotr.y(4,:)>0)<val(i)));
end;


[dummy,index]=min(res);
cut(3)=val(index)

clear z zt zr res;

% end_ of training

for rr= testset_runs,
  file= sprintf('%s%03d', subject, rr),
  load([data_dir file]);
  cnt.x= signal;
  
  if sum(trial==max(trial))<sum(trial==1),  %% delete incomplete last trials
    trial(find(trial==max(trial)))= -1;
  end

  pos= find(diff([-1; trial])>0) + 176;  %% time of feedback presentation
  mrk= struct('pos',pos', 'fs',cnt.fs);
  mrk.className= {'top','upper','lower','bottom'};

  cnt= proc_linearDerivation(cnt, csp_w);
  cnt.clab= csp_clab;

  epo= makeEpochs(cnt, mrk, [0 4000]);

  intervals=[750:250:2500; 1250:250:3000]';

clear feat1 feat2 feat prov

  for i=1:size(intervals,1)
    prov=proc_selectIval(epo,intervals(i,:));
    feat1=proc_fourierBandMagnitude(prov,dscr_band(1,:));
    feat2=proc_fourierBandMagnitude(prov,dscr_band(2,:));
    feat(i)=proc_catFeatures(feat1,feat2);
  end;

clear feat1 feat2 prov;

clear z zr zt predtargetpos;

for i=1:size(intervals,1); zr(i,:)=applyClassifier(feat(i),model,C); end;

th=3;
zt=zr;
zt(zr<-th)=-th;
zt(zr>th)=th;
%zt((z>=-th) & (z<=th))=0;

%z=sum(zr);
z=sum(zt);

predtargetpos(find(z<cut(1)))=1; 
predtargetpos(find((z<cut(2))&(z>=cut(1))))=2;
predtargetpos(find((z<cut(3))&(z>=cut(2))))=3;
predtargetpos(find(z>=cut(3)))=4;
 

  runnr= run(pos);
  trialnr= trial(pos);
  
  save_file= sprintf('%s%03dRESonline', subject, rr);
  save(save_file, 'runnr', 'trialnr', 'predtargetpos');
  
end

end 