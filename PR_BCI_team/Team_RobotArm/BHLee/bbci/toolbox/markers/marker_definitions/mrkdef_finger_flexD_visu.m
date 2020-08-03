function mrk= mrkdef_flexD(Mrk, file, opt)

classDef= {11, 12, 13, 21, 22, 23; 
           'flex fast (D)', 'flex med (D)', 'flex slow (D)', ...
           'ext fast (U)', 'ext med (U)', 'ext slow (U)'};
mrk= makeClassMarkers(Mrk, classDef,0,0);

classDef= {32, 33; 'prestimulus', 'stimulus off'};
mrk.stim= makeClassMarkers(Mrk, classDef,0,0);
