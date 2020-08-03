function mrk= mrkdef_flex_audi(Mrk, file, opt)

classDef= {4, 8;
           'ext', 'flex'};
mrk= makeClassMarkers(Mrk, classDef,0,0);

classDef= {[1 2 3 5 6 7]; 'no-go'};
mrk.click= makeClassMarkers(Mrk, classDef,0,0);
