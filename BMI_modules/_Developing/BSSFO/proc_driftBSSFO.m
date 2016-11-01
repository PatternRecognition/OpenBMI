function newSMC = proc_driftBSSFO( oldSMC )

newSMC = oldSMC;

% sigma = cov( (oldSMC.sample)' );
% sigma = var( (oldSMC.sample(2, :))' );
% R = chol( sigma );

% newSMC.sample = newSMC.sample + randn(2, newSMC.numBands)*R;
% newSMC.sample(1, :) = newSMC.sample(1, :) + randn(1, newSMC.numBands);
% newSMC.sample(2, :) = newSMC.sample(2, :) + randn(1, newSMC.numBands)*R;

for i=1:oldSMC.numBands
    if newSMC.selected( i ) == 0
        newSMC.sample(:, i) = abs( newSMC.sample(:, i) + randn(2, 1)*1 );
    %     newSMC.sample(2, i) = newSMC.sample(1, i) + 4;
        newSMC.sample(:, i) = opt_checkValidity( newSMC.sample(:, i) );
    end
end