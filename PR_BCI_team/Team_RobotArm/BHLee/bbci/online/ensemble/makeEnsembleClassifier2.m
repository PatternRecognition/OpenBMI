load([EEG_RAW_DIR 'subject_independent_classifiers/Lap_C3z4_LR.mat'])
load([BCI_DIR 'online/ensemble/filters.mat'])
load([BCI_DIR 'online/ensemble/Opt.mat'])
load([BCI_DIR 'online/ensemble/LSR_L1_w.mat'])
load([BCI_DIR 'online/ensemble/baseClassifiers.mat'])
Nchans=size(opt.clab,2);
% find out which filters we really need and the indices of the csp's:
w=mean(W,2);
%w_matrix=reshape(w,18,83);
%[a,b]=sort(w,'descend');
%loss=reshape(dat.loss,size(dat.loss,1)*size(dat.loss,2),1);
%loss(b(1:10))
ind=find(abs(w)>1e-6); %-6
length(ind)

[a,b]=ind2sub([18 83], ind);
%[b,a]=ind2sub([83 18], ind);
 
for i=1:5,dat.loss(b(i),a(i)),end
 
[a,inxa]=sort(a);
b=b(inxa);

for i=1:size(filterbank,1)
  Nfilter(i)=sum(a==i);
  filter_positions{i}=find(a==i);
end
%plot(Nfilter)
for i=1:83
  Nsubject(i)=sum(b==i);
  subject_positions{i}=find(b==i);
end
load baselineXvalLR.mat
dum=corrcoef(lossAuto,Nsubject);
%asd(jjj)=dum(2,1);
%end
%mean(asd)


% cls
cls.applyFcn='apply_separatingHyperplaneE';
cls.C=[];

dum=dat.csp{1,1};
CSP=zeros(length(ind),size(dum,2),size(dum,1));
cls.C.w=[];
cls.C.b=[];
cls.C.z0=w(ind)./sum(w(ind));
cls.C.z=w(ind)./sum(w(ind));
%cls.C.z=ones(length(ind),1);
cls.C.bias=0;
for j=1:length(ind)
  CSP(j,:,:)=dat.csp{b(j),a(j)}';
  size(CSP);
  
  cls.C.w=[cls.C.w; dat.classy{b(j),a(j)}.w];
  cls.C.b=[cls.C.b; dat.classy{b(j),a(j)}.b];
  size(cls.C.w);
  size(cls.C.b);
end
disp(sprintf('size of csp %i %i %i',size(CSP)))
disp(sprintf('size of cls.C.w %i %i',size(cls.C.w)))
disp(sprintf('size of cls.C.b %i %i',size(cls.C.b)))

csp=CSP;
clear CSP
j=1;

% size(a_array);
% dum0=a_array(a,:);
% dum1=b_array(a,:);
% clear a_array b_array
% a_array=dum0;
% b_array=dum1;
% size(a_array);

% csp will have size of:
%csp=

% need to change following variables:
% analyze, bbci, cls, cont_proc, feature


% analyze
analyze.fs=100;
analyze.band=filterbank;
analyze.filt_b=b_array;
analyze.filt_a=a_array;
analyze.spat_w=[];

%bbci can stay the way it is for now
bbci.Nfilter=Nfilter;
bbci.fs=100;
bbci.adaptation.running=1;
bbci.adaptation.policy='ensemble';
%bbci.adaptation.mrk_start=[1 2];
%bbci.adaptation.mrk_end=60;
bbci.adaptation.verbose= 0;
bbci.adaptation.alpha=0.05;
bbci.feedback='1d';
%need to a downsampling filter here:
Wps= [40 49]/bbci.fs*2;
[n, Ws]= cheb2ord(Wps(1), Wps(2), 3, 50);
[bbci.filt.b, bbci.filt.a]= cheby2(n, 50, Ws);


% cont_proc
cont_proc.clab=opt.clab;
cont_proc.procFunc={'online_filtBank','online_linearDerivationE'};
cont_proc.procParam={{b_array,a_array},{csp,Nfilter}};
% feature
feature.proc={'proc_variance','proc_logarithm'};
feature.proc_param={{},{}};
feature.cnt=1;
feature.ilen_apply=750;

save([EEG_RAW_DIR 'subject_independent_classifiers/ensemble_LR.mat'],'analyze','bbci','cont_proc','cls','feature')