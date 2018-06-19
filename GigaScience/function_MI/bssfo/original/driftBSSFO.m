function newSMC = driftBSSFO( oldSMC )

newSMC = oldSMC;

for i=1:oldSMC.numBands
    if newSMC.selected( i ) == 0
        newSMC.sample(:, i) = abs( newSMC.sample(:, i) + randn(2, 1)*1 );
        newSMC.sample(:, i) = checkValidityBSSFO( newSMC.sample(:, i) );
    end
end