function v= goodContourValues(mi, ma, ct)
% to be optimized

if ct<0,
  spacing= (ma-mi)/(-ct+1);
  gro= floor(log10(spacing));
  res= spacing*10^-gro;
  if res<1.5,
    resi= 1;
  elseif res<3.5,
    resi= 2;
  elseif res<7,
    resi= 5;
  else
    resi= 1;
    gro= gro+1;
  end
  ct= resi*10^gro;
end

v= ct*ceil(mi/ct):ct:ct*floor(ma/ct);
