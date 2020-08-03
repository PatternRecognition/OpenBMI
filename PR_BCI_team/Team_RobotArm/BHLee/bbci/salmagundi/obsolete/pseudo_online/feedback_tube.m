if ~isstruct(dsply), 
  dsply= struct('E',dsply);
end
if ~isfield(dsply, 'scale'), dsply.scale= 1; end
if ~isfield(dsply, 'tubePercent'), dsply.tubePercent=[5 10 15]; end
if ~isfield(dsply, 'facealpha'), dsply.facealpha=1; end
if ~isfield(dsply, 'color'), dsply.color= []; end

ma= feedback_opt.integrate;

if exist('dtct','var'),
  error('not implemented');
end

dscr_out_ma= movingAverageCausal(dscr_out, ma);


if ~exist('dtct','var'),

iv= getIvalIndices(dsply.E([1 end]), feedback_opt.fs);
mrk_test= mrk_selectEvents(mrk, first_test_event:length(mrk.pos)-1);
mrk_test.pos= mrk_test.pos - test_begin + 1;
mrk_test.pos= round(mrk_test.pos/mrk_test.fs*feedback_opt.fs);
mrk_test.fs= feedback_opt.fs;
test_cl= {find(mrk_test.y(1,:)), find(mrk_test.y(2,:))};
nTestClasses= length(test_cl);
nEpochs= length(mrk_test.pos);
nShifts= length(iv);
outTraces= zeros(nShifts, nEpochs);
testTube= zeros(nShifts, 3+2*length(dsply.tubePercent), nTestClasses);

for ei= 1:nEpochs,
  outTraces(:,ei)= dscr_out_ma(mrk_test.pos(ei)+iv);
end

for is= 1:nShifts, 
  for ic= 1:nTestClasses,
    outCl= outTraces(is, test_cl{ic});
    testTube(is, :, ic)= fractileValues(outCl(:), dsply.tubePercent);
  end
end

clf;
time_line= linspace(dsply.E(1), dsply.E(end), nShifts);
if dsply.facealpha==1,
  hp= plotTube(testTube, time_line);
else
  hp= plotTubeNoFaceAlpha(testTube, time_line, dsply.color);
end

else

  error('not implemented');
  
end  %% if-else isempty(dtct)

figure(1)
