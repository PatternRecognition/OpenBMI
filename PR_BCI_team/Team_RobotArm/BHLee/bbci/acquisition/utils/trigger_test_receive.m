state= acquire_bv(100, general_port_fields.bvmachine);

nRounds= 10;
counter= zeros(1, 255);
while sum(counter)<255*nRounds,
  [dmy,bn,mp,mt,md]= acquire_bv(state);
  for mm= 1:length(mt),
    trig= str2num(mt{mm}(2:4));
    counter(trig)= counter(trig)+1;
    if mod(sum(counter),255)==0,
      counter
    end
  end
end
acquire_bv('close');
