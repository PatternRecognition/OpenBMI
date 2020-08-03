function Ind = getNotchIndices(notch, fs, N)

Freq= (0:N/2)*fs/N;

% search for multiples of the notch frequency:

nf = Freq/notch;

dis = abs(nf-round(nf));

Ind = find(dis<eps);
Ind = Ind(2:end);

if isempty(Ind)
  error('The window length does not match the notch-frequency. It has to be a multiple of it.');
  
end

