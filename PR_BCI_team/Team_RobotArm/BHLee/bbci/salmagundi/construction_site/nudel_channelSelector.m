%% fv, model, opt_xv

fprintf('* incremental *\n');
nChans= size(fv.x,2);
l1= zeros(nChans, 1);
for cc= 1:nChans,
  ff= proc_selectChannels(fv, cc);
  l1(cc)= xvalidation(ff, model, opt_xv, ...
                        'out_prefix',sprintf('%s: ',fv.clab{cc}));
end
[mm,mi]= min(l1);
fprintf('\nbest result: %s: %.1f%%\n', fv.clab{mi}, 100*mm);

fprintf('* decremental *\n');
nChans= size(fv.x,2);
l2= zeros(nChans, 1);
for cc= 1:nChans,
  ff= proc_selectChannels(fv, 'not', fv.clab{cc});
  l2(cc)= xvalidation(ff, model, opt_xv, ...
                        'out_prefix',sprintf('%s: ',fv.clab{cc}));
end
[mm,mi]= max(l2);
fprintf('\nworst result: %s: %.1f%%\n', fv.clab{mi}, 100*mm);

[so,si]= sort(l1-l2);
fv.clab(si)
