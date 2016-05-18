function CSPFilter = spatial_filtering( x_flt, oldSMC, numPatterns, verbose )

if verbose == 1
    fprintf( '\tSpatial Filtering...\n\t\t' );
end

CSPFilter = cell( 1, oldSMC.numBands );
for i=1:oldSMC.numBands
    if verbose == 1
        if mod(i, 5) == 0
            fprintf( '%d', i );
        else
            fprintf( '.' );
        end
        if mod(i, 100) == 0
            fprintf( '\n' );
        end
    end
    
    D1 = x_flt{1, i};
    D2 = x_flt{2, i};
    [W, D] = myTrainCSP( D1, D2 );
%     [W, D] = myCSPFilter( D1, D2 );
    CSPFilter{i}.W = W( :, [1:numPatterns, end-numPatterns+1:end] );
    Dd = diag(D);
    CSPFilter{i}.D = Dd([1:numPatterns, end-numPatterns+1:end]);
end
fprintf( '\n' );
