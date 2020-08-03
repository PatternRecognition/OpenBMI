function [epo, dtct, dscr, dtct_wnd, dscr_wnd]= ...
    bciDal_train(file, dtct, dscr, train_frac)
%function [epo, dtct, dscr, dtct_wnd, dscr_wnd]= ...
%    bciHa_train(train_file, dtct, dscr, <train_frac/nTrains>)

if ~exist('train_frac','var'), train_frac=1; end

[cnt, mrk]= loadProcessedEEG(file);
if train_frac<=1,
  nTrains= floor(length(mrk.pos)*train_frac);
else
  nTrains= train_frac;
end

mrk_train= mrk_selectEvents(mrk, 1:nTrains);

epo= makeEpochs(cnt, mrk_train, dtct.ival, dtct.motoJits);
nMotos= size(epo.y, 2);
epo_no_moto= makeEpochs(cnt, mrk_train, ...
                          dtct.ival+dtct.shift, dtct.nomotoJits);
noMotos= size(epo_no_moto.y, 2);
epo= proc_appendEpochs(epo, epo_no_moto);
%epo.y= [repmat([0;1], 1, nMotos) repmat([1;0], 1, noMotos)];
epo.y= zeros(3, size(epo.y,2));
%% labels for no motor events
epo.y(1,nMotos+1:end)= 1;
%% labels for l/r motor events
epo.y(2:3,1:nMotos)= repmat(mrk_train.y, [1 length(dtct.motoJits)]);  
clear epo_no_moto
dtct= fb_run_train(epo, mrk_train, dtct);
dtct_wnd= copyStruct(epo, 'x','y','nJits');


epo= makeEpochs(cnt, mrk_train, dscr.ival, dscr.jits);
dscr= fb_run_train(epo, mrk_train, dscr);
dscr_wnd= copyStruct(epo, 'x','y','nJits');
