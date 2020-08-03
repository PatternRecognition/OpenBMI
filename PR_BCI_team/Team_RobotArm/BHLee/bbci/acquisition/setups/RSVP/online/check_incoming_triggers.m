state= acquire_bv(100, general_port_fields.bvmachine);

nRounds= 100;
counter= zeros(1, 30);
while sum(counter)<255*nRounds,
  [dmy,bn,mp,mt,md]= acquire_bv(state);
  for mm= 1:length(mt),
    tt= str2num(mt{mm}(2:4));
    if tt>=31 & tt<=100,
      trig= mod(tt-31,40)+1;
      counter(trig)= counter(trig)+1;
      if mod(sum(counter),30)==0,
        counter
      end
    end
  end
end
acquire_bv('close');
