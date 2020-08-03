function mrk= mrkdef_oddball_auditory(Mrk, file, opt)

classDef= {1, 2;
           'std','dev'};
mrk= makeClassMarkers(Mrk, classDef,0,0);

classDef= {10; 'stimulus off'};
mrk.stim= makeClassMarkers(Mrk, classDef,0,0);
