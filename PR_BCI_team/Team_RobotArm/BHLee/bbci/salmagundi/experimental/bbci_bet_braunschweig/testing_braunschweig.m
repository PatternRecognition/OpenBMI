%setup_bbci_bet_braunschweig
%cd([BCI_DIR 'bbci_bet_braunschweig'])
%makeacquire_bv_braunschweig

%return



bs= acquire_bv_braunschweig(100, 'brainamp')
[currData, block] = acquire_bv_braunschweig(bs);

dd= zeros(20000, size(currData,2));
k= 0;
t0= clock;
bsz= [];
while etime(clock,t0)<10.0,
  [currData, block] = acquire_bv_braunschweig(bs);
  nSamples= size(currData,1);
  dd(k+[1:nSamples],:)= currData;
  k= k+nSamples;
  bsz= [bsz nSamples];
end
fprintf('estimated fs= %.1fHz, mean blocksize= %.1f\n', k, mean(bsz));
plot(dd(1:k,:));

bbciclose;

return



