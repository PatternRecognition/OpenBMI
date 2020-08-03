function prepareSelfpacedfeet(file, filterList)
%prepareSelfpacedfeet(file, filterList)

if ~exist('filterList', 'var'),
  filterList= {'raw','cut50'};
end

classDef= {'V','N'; 'left foot', 'right foot'};
blockingTime= 250;
blockingDoubles= 55;
prepareSelfpaced(file, filterList, classDef, 'small', ...
                 blockingTime, blockingDoubles);
