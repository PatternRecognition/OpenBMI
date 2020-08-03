

%% Variance loss 
[myseq, stat] = pseudoRandMatrix(5,6,5,15,1,1);
mymin=stat;
mymax=stat;

for i=1:30
    [myseq, stat] = pseudoRandMatrix(5,6,5,15,1,1);
    if stat.var_overallSeq < mymin.var_overallSeq
        mymin=stat
    end
    if stat.var_overallSeq > mymax.var_overallSeq
        mymax=stat
    end 
end
  
mymin.var_overallSeq
mymax.var_overallSeq

clims=[0 max(max(mymax.count))]
figure; imagesc(mymin.count,clims) ; colorbar
figure; imagesc(mymax.count,clims) ; colorbar
