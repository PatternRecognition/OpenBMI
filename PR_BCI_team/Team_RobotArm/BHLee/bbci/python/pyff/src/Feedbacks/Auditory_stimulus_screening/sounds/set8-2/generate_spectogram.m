flist = {...
         '1.wav' ...
	 '2.wav' ...
	 '3.wav' ...
         '4.wav' ...
	 '5.wav' ...
	 '6.wav' ... 
         '7.wav' ...
	 '8.wav' ...
	 '9.wav' ... 
        };

iix = [1 1 2 1 1 2 1 1 2];

figure;
for ii = 1:9
  subplot(3,3,ii)
  fname = flist{ii};
  [aa fs] = wavread([fname]);
  aa = aa(:,iix(ii));
  logf = [2:.01:5];
  F = power(10,logf);

  [S F T] = spectrogram(aa,256,250,F,fs);
  %[S F T] = specgram(aa, 256, fs);
  %specgram(aa, 256, fs);
  imagesc(T,F,abs(S));
  if ii < 7
    set(gca, 'XTick', [])
    set(gca, 'XTickLabel', [])
  else
    xlabel('time in sec')
  end
  if mod(ii,3) == 1
    ylabel('log f')
  end
  set(gca, 'YTick', [])
  set(gca, 'YTickLabel', [])

end

%printFigure(['/home/konrad/Uni/bachelorarbeit/'], 'format', 'pdf')
