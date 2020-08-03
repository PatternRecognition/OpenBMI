EEG_AUGCOG_DIR= ['/home/neuro/data/AUGCOG/'];

augcog_file= 'augcog_database_season1.txt';

[file, carf, audio, visual, calc]= ...
    textread(augcog_file, '%s%s%s%s%s', 'delimiter',',');
file= deblank(file);
carf = deblank(carf);
audio = deblank(audio);
visual = deblank(visual);
calc = deblank(calc);

augcog= struct('file', file, ...
               'carf', carf,...
               'audio', audio,...
               'visual',visual,...
               'calc',calc);
