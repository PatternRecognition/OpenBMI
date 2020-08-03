%% cnt, mkr, dtct, dscr, feedback_opt, nTrains

mrk_train= pickEvents(mrk, 1:nTrains);

epo= makeSegments(cnt, mrk_train, dtct.ival, dtct.motoJits);
nMotos= size(epo.y, 2);
epo_no_moto= makeSegments(cnt, mrk_train, ...
                          dtct.ival+dtct.shift, dtct.nomotoJits);
noMotos= size(epo_no_moto.y, 2);
epo= proc_appendEpochs(epo, epo_no_moto);
epo.y= [repmat([0;1], 1, nMotos) repmat([1;0], 1, noMotos)];
clear epo_no_moto
dtct= fb_run_train(epo, mrk_train, dtct);
dtct_wnd= copyStruct(epo, 'x','y','nJits');


epo= makeSegments(cnt, mrk_train, dscr.ival, dscr.jits);
dscr= fb_run_train(epo, mrk_train, dscr);
dscr_wnd= copyStruct(epo, 'x','y','nJits');
