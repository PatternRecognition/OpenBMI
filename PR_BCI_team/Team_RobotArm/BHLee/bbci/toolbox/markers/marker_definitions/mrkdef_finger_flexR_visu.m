function mrk= mrkdef_flexR(Mrk, file, opt)

classDef= {11, 12, 13, 21, 22, 23; 
           'ext fast (L)', 'ext med (L)', 'ext slow (L)', ...
           'flex fast (R)', 'flex med (R)', 'flex slow (R)'};
mrk= makeClassMarkers(Mrk, classDef,0,0);

classDef= {32, 33; 'prestimulus', 'stimulus off'};
mrk.stim= makeClassMarkers(Mrk, classDef,0,0);
