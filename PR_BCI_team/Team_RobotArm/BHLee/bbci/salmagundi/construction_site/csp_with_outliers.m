label = 'Gabriel_03_04_01_imag_LR'
load(['/home/data/BCI/bbciEpoched/' label]);
epo=fv;
[epoTr, epoTe]=proc_splitSamples(epo, [1 1]);
[fv, W, la, A]=proc_csp3(epoTr, 'patterns', 3);
fv=proc_logarithm(proc_variance(fv));
fvTe=proc_featureCSP(epoTe, W);
ixOutlTr=find(max(shiftdim(fv.x))>1)
ixOutlTe=find(max(shiftdim(fvTe.x))>1)

memo =[];
proc = {'none', 'clean', 'add'}
for i=1:length(proc)
  switch (proc{i})
   case 'none',
    fv = epoTr;
   case 'clean',
    fv = proc_selectEpochs(epoTr, 'not', ixOutlTr);
   case 'add',
    fv = epoTr;
    X = fv.x(:,:,ixOutlTr);
    Y = [0, 1; 1 0] * fv.y(:,ixOutlTr);
    fv.x = cat(3, fv.x, X);
    fv.y = [fv.y, Y];
  end
  [fv, W, la, A] = proc_csp3(fv, 'patterns', 1);
  fv=proc_logarithm(proc_variance(fv));
  C = trainClassifier(fv, 'LDA');
  fvTe = proc_featureCSP(epoTe, W);
  out = applyClassifier(fvTe, 'LDA', C);
  loss = mean(loss_0_1(fvTe.y, out))
  loss_at = loss_0_1(fvTe.y(:,ixOutlTe), out(ixOutlTe))
  
  memo = [memo archive('W','la','A','C','out','loss','loss_at')];
end





