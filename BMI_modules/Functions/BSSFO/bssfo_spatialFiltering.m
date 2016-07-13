function CSPFilter = bssfo_spatialFiltering( x_flt, oldSMC, numPatterns, verbose )

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
    
    X1 = x_flt{1, i};
    X2 = x_flt{2, i};
    
    %% Here is basic CSP filtering.
    [nc1, ns1, nt1] = size( X1 );
    [nc2, ns2, nt2] = size( X2 );
    
    XX1 = reshape( permute(X1, [2 3 1]), [ns1*nt1, nc1]);
    S1 = cov(XX1(:,:));
    
    XX2 = reshape( permute(X2, [2 3 1]), [ns2*nt2, nc2]);
    S2 = cov(XX2(:,:));
    
    [W,D] = eig(S1, S1+S2);
    %%
    CSPFilter{i}.W = W( :, [1:numPatterns, end-numPatterns+1:end] );
    Dd = diag(D);
    CSPFilter{i}.D = Dd([1:numPatterns, end-numPatterns+1:end]);
end
fprintf( '\n' );
