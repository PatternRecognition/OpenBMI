function features = bssfo_featureExtraction( x_flt, CSP, oldSMC, verbose )
% x_flt: cell(2, numsamples) : 2-class
%           x_flt{1, i} = [nChannels, nSamples, nTrials]

if verbose == 1
    fprintf( '\tFeature Extraction...\n\t\t' );
end

[nclass, nparticles] = size( x_flt );
features = cell( nclass, nparticles );

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
    
    for k=1:nclass
        [nc, ns, nt] = size( x_flt{k, i} );
        dat = reshape( x_flt{k, i}, [nc, ns*nt] );
        dat=func_projection(dat', CSP{i}.W);    
        dat = reshape( dat, [ns, nt, size(dat, 2)] );
        dat = permute( dat, [1 3 2] );
        features{k, i} = squeeze( log(var(dat, 0, 1)) ); % log-variance feature
    end
end
fprintf( '\n' );