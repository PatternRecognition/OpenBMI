function out= proc_peakSearch(fv, aa, bb, varargin)

opt= propertylist2struct(varargin{:});
opt= set_defaults(opt, 'intersect_with_band', []);

for ai= 1:length(aa),
  for bi= 1:length(bb),
    ival= [aa(ai) bb(bi)];
    if isempty(opt.intersect_with_band),
      intersect_ival= ival;
    else
      intersect_ival= [max(ival(1),opt.intersect_with_band(1)), min(ival(2),opt.intersect_with_band(2))];
    end
    if intersect_ival(1)>=intersect_ival(2),
      continue;
    end
    pa= proc_peakArea(fv, intersect_ival, 'interpolation_ival',ival, 'relative',1);
%    pa.x= pa.peak;
    if ai==1 & bi==1,
      out= rmfield(pa, 'peak');
      out.ival= repmat(intersect_ival, [length(pa.x), 1]);
      out.interpolation_ival= repmat(ival, [length(pa.x), 1]);
    end
    idx= find( pa.x > out.x );
    out.x(idx)= pa.x(idx);
    out.ival(idx,:)= repmat(intersect_ival, [length(idx), 1]);
    out.interpolation_ival(idx,:)= repmat(ival, [length(idx), 1]);
    out.peak_time(idx)= pa.peak_time(idx);
  end
end
