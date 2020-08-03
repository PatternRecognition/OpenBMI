function mrk= mrkdef_noflex_audi(Mrk, file, opt)

classDef= {4, 8;
           'low tone (ext)', 'high tone (flex)'};
mrk= makeClassMarkers(Mrk, classDef,0,0);

classDef= {[1 2 3 5 6 7]; 'soft (no-go)'};
mrk.click= makeClassMarkers(Mrk, classDef,0,0);
