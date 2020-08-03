% test the function getClassifier.m
% Make a buffer for two channels, length 3

% initialize the feature vectors
feature.cnt = 1;
feature.ilen_apply = 1000;
feature.proc = {'proc_linearDerivation','proc_jumpingMeans'};
feature.proc_param = {{[1;1]}, {100}};
opt = struct('fs',100);
cont_proc = struct;
cont_proc.clab = {'ch1','ch2'};

getFeature('init',feature,opt,cont_proc);

% initialize the ring buffer and fill it with data.
storeContData('init',1,2,'ringBufferSize',30000,'fs',100);
pos = 0;
for i = 1:100
  pos = pos+randn;
  storeContData('append',1,[pos+rand*0.1 pos+rand*0.1;...
		    pos+rand*0.1 pos+rand*0.1;...
		    pos+rand*0.1 pos+rand*0.1;...
		    pos+rand*0.1 pos+rand*0.1]);
end
%initialize the feature function.
getFeature('init',feature,struct,cont_proc);


% Initialize the classifier!!!
cls = struct('condition',nan);
cls.condition_param = [];
cls.fv = [1];
cls.applyFcn = 'apply_separatingHyperplane';
cls.C.w = eye(length(cls.fv));
cls.C.b = 0

opt = struct;
% this should evaluate the classifier.
cls = getClassifier('init',cls,opt);
out = getClassifier('apply',1500,cls)

% this should not evaluate it.
cls.condition = 'F(0);';
cls = getClassifier('init',cls,opt);
out = getClassifier('apply',1500,cls)

% this should also handle several classifiers.
cls(2) = cls;
cls(2).condition = ['F(cl{1}>10);'];
cls = getClassifier('init',cls,opt);
out = getClassifier('apply',1500,cls)

cls(2).condition = ['F(cl{1}<10);'];
cls = getClassifier('init',cls,opt);
out = getClassifier('apply',1500,cls)

% cleanup:
storeContData('cleanup');