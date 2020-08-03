function [bInd, Freq]= getBandIndices(band, fs, N)
%[bInd, Freq]= getBandIndices(band, fs, N)
%
% IN  band  - [lower upper] limit of band in Hz,
%             band may also be a sequence of bands, one in each row
%     fs    - sampling frequency
%     N     - window width in samples
%
%     bInd  - indices of the bins corresponding to the given band
%     Freq  - center frequencies of the bins

sb= size(band);
if isequal(sb, [2 1]), band= band'; end
if sb(1)>1,
  bInd= [];
  for ib= 1:sb(1);
    [bi, Freq]= getBandIndices(band(ib,:), fs, N);
    bInd= cat(2, bInd, bi);
  end
  return
end

  
Freq= (0:N/2)*fs/N;

if isempty(band)
  bInd= 1:N/2+1;
else
  ibeg= max(find(Freq<=band(1)));
  iend= min([find(Freq>=band(2)) length(Freq)]);
  bInd= ibeg:iend;
end

Freq= Freq(bInd);
