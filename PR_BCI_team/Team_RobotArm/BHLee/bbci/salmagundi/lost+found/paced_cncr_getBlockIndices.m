function mrk= paced_cncr_getBlockIndices(mrk, classes1, classes2)
%mrk= paced_cncr_getBlockIndices(mrk, <classes1, classes2>)
%
% in the paced_cncr paradigm epochs of the classes {'* click','*no-click'}
% are acquired in blockwise alternation with epochs of the class
% 'rest'. this function add a field 'bidx' to the marker structure
% which hold block indices for each marker.
% if epochs are generated from this marker structure, a subsequent
% doXvalidationPlus will respect blocks in the train/test set splittings.
%
% defaults: classes1= {'* click','*no-click'}, classes2= {'rest'}

if ~exist('classes1', 'var'),
   classes1= {'* click','*no-click'};
end
if ~exist('classes2', 'var'),
  classes2= {'rest'};
end

cl1= getClassIndices(mrk, classes1);
cl2= getClassIndices(mrk, classes2);
ccc= zeros(1, size(mrk.y,1));
ccc(cl1)= 1;
ccc(cl2)= 2;

bidx= zeros(1,length(mrk.pos));
bb= 0;
lc= 0;
for mm= 1:length(mrk.pos),
  ac= ccc*mrk.y(:,mm);
  if ac~=lc,
    bb= bb+1;
  end
  lc= ac;
  bidx(mm)= bb;
end
mrk.bidx= bidx;
mrk.indexedByEpochs= cat(2, mrk.indexedByEpochs, {'bidx'});
