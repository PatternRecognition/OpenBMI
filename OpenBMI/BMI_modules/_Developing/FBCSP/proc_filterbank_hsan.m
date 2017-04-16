function [ dat,b,a ] = proc_filterbank_hsan( dat, order, band )
    [b,a]=butters(order, band/dat.fs*2);
    dat = proc_filterbank(dat, b, a);
    
end

