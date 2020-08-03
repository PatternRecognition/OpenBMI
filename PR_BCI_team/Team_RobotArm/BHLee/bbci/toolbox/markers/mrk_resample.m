function mrk= mrk_resample(mrk, fs)

mrk.pos= round(mrk.pos/mrk.fs*fs);
mrk.fs= fs;
