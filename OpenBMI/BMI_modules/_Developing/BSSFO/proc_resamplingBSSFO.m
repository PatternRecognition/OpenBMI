% File Name: resampling.m
% Author: Heung-Il Suk
% Cite: H.-I. Suk and S.-W. Lee, "A Novel Bayesian Framework for Discriminative 
% Feature Extraction in Brain-Computer Interfaces," IEEE Trans. on PAMI,
% 2012. (Accepted)

function newBSSFO = proc_resamplingBSSFO( oldBSSFO )

accWeight = cumsum( oldBSSFO.weight );

for i=1:oldBSSFO.numBands
    oldBSSFO.selected( i ) = 0;
end

newBSSFO = oldBSSFO;
for i=1:oldBSSFO.numBands
    idx = find( accWeight > rand );
    a(i)=idx(1);
    newBSSFO.sample(:, i) = oldBSSFO.sample( :, idx(1) );
    if oldBSSFO.selected( idx(1) ) == 0
        oldBSSFO.selected( idx(1) ) = 1;
        newBSSFO.selected( i ) = 1;   % keep current state, do not add random noise
    end
end

newBSSFO.weight = ones(newBSSFO.numBands, 1)./ newBSSFO.numBands;