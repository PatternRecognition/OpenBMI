load('/home/bbci/data/bbciRaw/subject_independent_classifiers/Lap_C3z4_LR.mat')
load filters
load Opt
load LSR_L1_w
load baseClassifiers
Nchans=size(opt.clab,2);
% find out which filters we really need and the indices of the csp's:
w=mean(W,2);
%figure;imagesc(reshape(w,[18, 83])')
%ind=find(abs(w)>1e-2);

ind=find(abs(w)>1e-4);

%it should be that:
%ind=find(abs(w)>1e-6);

%w=w(ind);
[a,b]=ind2sub([18 83], ind);


% cls
cls.applyFcn='apply_separatingHyperplaneE';
cls.C=[];

dum=dat.csp{1,1};
CSP=zeros(length(ind),size(dum,2),size(dum,1));
cls.C.w=[];
cls.C.b=[];
cls.C.z=w(ind)./sum(w(ind));
for j=1:length(ind)
  CSP(j,:,:)=dat.csp{b(j),a(j)}';
  size(CSP)
  
  cls.C.w=[cls.C.w; dat.classy{b(j),a(j)}.w];
  cls.C.b=[cls.C.b; dat.classy{b(j),a(j)}.b];
  size(cls.C.w);
  size(cls.C.b);
end
disp(sprintf('size of csp %i %i',size(CSP)))
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
bbci.fs=100;
bbci.adaptation.running=0;
%bbci.adaptation.policy='pmean';
% cont_proc
cont_proc.clab=opt.clab;
cont_proc.procFunc={'online_filtBank','online_linearDerivation'};
cont_proc.procParam={{b_array,a_array},{csp}};
% feature
feature.proc={'proc_variance','proc_logarithm'};
feature.proc_param={{},{}};
feature.cnt=1;
feature.ilen_apply=1000;

save('/home/bbci/data/bbciRaw/subject_independent_classifiers/ensemble_LR.mat','analyze','bbci','cont_proc','cls','feature')