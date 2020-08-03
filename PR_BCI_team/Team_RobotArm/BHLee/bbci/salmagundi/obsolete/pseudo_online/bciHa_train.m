function [epo, dscr, dscr_wnd, dtct, dtct_wnd]= ...
    bciHa_train(file, dscr, dtct, train_frac)
%[epo, dscr, dscr_wnd, dtct, dtct_wnd]= ...
%      bciHa_train(file, dscr, dtct, <train_frac>)

if ~exist('train_frac','var'), train_frac=1; end

[cnt, mrk]= loadProcessedEEG(file);
if train_frac<=1,
  nTrains= floor(length(mrk.pos)*train_frac);
else
  nTrains= train_frac;
end

mrk_train= mrk_selectEvents(mrk, 1:nTrains);


epo= makeEpochs(cnt, mrk_train, [-dscr.ilen 0], dscr.jit);
dscr= fb_run_train(epo, mrk_train, dscr);
epo= makeEpochs(cnt, mrk_train, [-dscr.ilen_apply 0]);
dscr_wnd= copyStruct(epo, 'x','y','bidx','jit');


if ~isempty(dtct),
  epo= makeEpochs(cnt, mrk_train, [-dtct.ilen 0], dtct.motoJit);
  nMotos= size(epo.y, 2);
  epo_no_moto= makeEpochs(cnt, mrk_train, ...
                          [-dtct.ilen 0], dtct.nomotoJit);
  noMotos= size(epo_no_moto.y, 2);
  epo= proc_appendEpochs(epo, epo_no_moto);
  %epo.y= [repmat([0;1], 1, nMotos) repmat([1;0], 1, noMotos)];
  epo.y= zeros(3, size(epo.y,2));
  %% labels for no motor events
  epo.y(1,nMotos+1:end)= 1;
  %% labels for l/r motor events
  epo.y(2:3,1:nMotos)= repmat(mrk_train.y, [1 length(dtct.motoJit)]);  
  clear epo_no_moto
  dtct= fb_run_train(epo, mrk_train, dtct);
  dtct_wnd= copyStruct(epo, 'x','y','bidx','jit');
end
