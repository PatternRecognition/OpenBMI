% Test cases for standard post-processing

% Only the following fields are actually used:
cls = struct('integrate', {1 2 3}, 'bias', {0 -100 100}, 'scale', {1 0.5 2});

% This must raise an error: 
standardPostProc('init', cls, 'maxBufLength', 1);
% This should work
standardPostProc('init', cls, 'maxBufLength', 5);

out = {1 1 1};
out2 = standardPostProc('apply', cls, out)
% out2{1} should be 1 (unchanged)
% out2{2} should be (1-100)*0.5 = -49.5
% out2{3} should be (1+100)*2 = 202
out = {2 2 2};
out2 = standardPostProc('apply', cls, out)
% out2{1} should be 2 (unchanged)
% out2{2} should be ((1+2)/2-100)*0.5 = -48.25
% out2{3} should be ((1+2)/2+100)*2 = 203
out = {3 3 3};
out2 = standardPostProc('apply', cls, out)
% out2{1} should be 3 (unchanged)
% out2{2} should be ((2+3)/2-100)*0.5 = -48.75
% out2{3} should be ((1+2+3)/3+100)*2 = 204

standardPostProc('cleanup');

cls(1).integrate = 1.7;
% This must raise an error: 
standardPostProc('init', cls, 'maxBufLength', 5);
