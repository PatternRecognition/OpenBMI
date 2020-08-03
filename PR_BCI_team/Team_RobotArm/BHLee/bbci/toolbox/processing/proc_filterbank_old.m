function cnt_flt= proc_filterBank(cnt, band_list, filtOrder)
%dat= proc_filterBank(dat, band, filtOrder)
%
% apply digital (FIR or IIR) filter
%
% IN   dat   - data structure of continuous or epoched data
%      band  - [nBands 2]-sized array of filter banks
%
% OUT  dat      - updated data structure

for bb= 1:size(band_list,1),
  [b,a]= butter(filtOrder, band_list(bb,:)/cnt.fs*2);
  band_str= sprintf(' [%g %g]', band_list(bb,:));
  if bb==1,
    cnt_flt= proc_filt(cnt, b, a);
    cnt_flt.clab= apply_cellwise(cnt.clab, 'strcat', band_str);
  else
    cnt_tmp= proc_filt(cnt, b, a);
    cnt_tmp.clab= apply_cellwise(cnt.clab, 'strcat', band_str);
    cnt_flt= proc_appendChannels(cnt_flt, cnt_tmp);
  end
end
