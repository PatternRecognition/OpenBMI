function mrk= mrkdef_flexU(Mrk, file, opt)

classDef= {11, 12, 13, 21, 22, 23; 
           'ext fast (D)', 'ext med (D)', 'ext slow (D)', ...
           'flex fast (U)', 'flex med (U)', 'flex slow (U)'};
mrk= makeClassMarkers(Mrk, classDef,0,0);

classDef= {32, 33; 'prestimulus', 'stimulus off'};
mrk.stim= makeClassMarkers(Mrk, classDef,0,0);
