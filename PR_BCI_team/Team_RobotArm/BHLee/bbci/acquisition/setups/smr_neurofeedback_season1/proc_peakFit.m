function out= proc_peakFit(fv, aa, bb)

for ai= 1:length(aa),
  for bi= 1:length(bb),
    ival= [aa(ai) bb(bi)];
    pa= proc_peakArea(fv, ival);
    pa.x= pa.peak;
    if ai==1 & bi==1,
      out= pa;
      out.ival= repmat(ival, [length(pa.x), 1]);
    end
    idx= find( pa.x > out.x );
    out.x(idx)= pa.x(idx);
    out.ival(idx,:)= repmat(ival, [length(idx), 1]);
  end
end
