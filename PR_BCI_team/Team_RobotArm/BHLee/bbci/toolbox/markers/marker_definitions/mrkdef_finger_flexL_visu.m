function mrk= mrkdef_flexL(Mrk, file, opt)

classDef= {11, 12, 13, 21, 22, 23; 
           'flex fast (L)', 'flex med (L)', 'flex slow (L)', ...
           'ext fast (R)', 'ext med (R)', 'ext slow (R)'};
mrk= makeClassMarkers(Mrk, classDef,0,0);

classDef= {32, 33; 'prestimulus', 'stimulus off'};
mrk.stim= makeClassMarkers(Mrk, classDef,0,0);
