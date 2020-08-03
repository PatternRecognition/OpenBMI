function prepareSelfpacedfeetmulti(file, filterList)
%prepareSelfpacedfeetmulti(file, filterList)

if ~exist('filterList', 'var'),
  filterList= {'raw','cut50'};
end

classDef= {'V','N'; 'left foot', 'right foot'};
blockingTime= 250;
simult= 50;
prepareSelfpacedmulti(file, filterList, classDef, 'small', ...
                      blockingTime, simult);
