function [fv, nBlocks, nClasses, sii]= proc_sortWrtStimulus(fv)

nClasses= max(fv.stimulus);
nBlocks= length(fv.stimulus)/nClasses;
A= reshape(fv.stimulus,[nClasses nBlocks]);
[so,si]= sort(A);
sii= si+repmat(0:nClasses:length(fv.stimulus)-1, nClasses, 1);
fv= proc_selectEpochs(fv, sii(:), 'removevoidclasses',0);
