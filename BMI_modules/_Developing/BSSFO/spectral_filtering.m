function x_flt = spectral_filtering( X1, X2, fs, oldSMC, verbose )

[nch1, ns1, nt1] = size( X1 );
XX1 = reshape( X1, [nch1, ns1*nt1] );

if ~isempty(X2)
    [nch2, ns2, nt2] = size( X2 );
    XX2 = reshape( X2, [nch2, ns2*nt2] );
    nclass = 2;
else
    nclass = 1;
end
x_flt = cell( nclass, oldSMC.numBands );

% Spectral Filtering
if verbose == 1
    fprintf( '\tSpectral Filtering...\n\t\t' );
end
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
    
    band = oldSMC.sample( :, i );
%     band(2) = band(1) + band(2);
    [b, a] = butter( 5, band/(fs/2), 'bandpass' );
    temp = filter( b, a, XX1' );
%     [b, a] = butter( 5, band(2)/(fs/2), 'low' );
%     temp = filter( b, a, XX1' );
%     [b, a] = butter( 5, band(1)/(fs/2), 'high' );
%     temp = filter( b, a, temp );

    temp = reshape( temp, [ns1, nt1, nch1] );
    x_flt{1, i} = permute( temp, [3 1 2] );
    if ~isempty( X2 )
        temp = filter( b, a, XX2' );
        temp = reshape( temp, [ns2, nt2, nch2] );
        x_flt{2, i} = permute( temp, [3 1 2] );
    end
end
if verbose == 1
    fprintf( '\n' );
end
