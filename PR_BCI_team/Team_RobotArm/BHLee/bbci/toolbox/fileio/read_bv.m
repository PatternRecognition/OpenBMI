function read_bv
% read_bv - read data from an eeg-File
%
% SYNOPSIS
%    data = read_bv(file, HDR, OPT); 
%
% ARGUMENTS
%                file - Name of EEG file (.eeg) is appended)
%                HDR  - Information about the file (read from the *.vhdr header file)
%                   .fs      - Sampling rate
%                   .nChans  - Number of channels
%                   .nPoints - Number of data points in the file (optional)
%                   .scale   - Scaling factors for each channel
%                   .endian  - Byte ordering: 'l' little or 'b' big
%                OPT  - Struct with following fields
%                   .chanidx         - Indices of the channels that are to be read
%                   .fs              - Down sample to this sampling rate
%                   .filt_b          - Filter coefficients of IIR filter 
%                                      applied to raw data (b part)
%                                      (optional)
%                   .filt_a          - Filter coefficients of IIR filter
%                                      applied to raw data (a part)
%                                      (optional)
%                   .filt_subsample  - Filter coefficients of FIR filter
%                                      used for sub sampling (optional)
%                   .data            - A matrix where the data is stored 
%                                      (optional)
%                   .dataPos         - The position in the matrix   
%                                      [dataStart dataEnd fileStart
%                                      fileEnd](optional) 
%
%      The filter parts of the OPT structure are optional fields.
%      The default for the filt_subsample is a filter which takes the last
%      value of filtered block e.g. [0 ... 0 1]
%
%      With opt.data and opt.dataPos read_bv can write directly to a
%      matrix. dataPos is an optional value for opt.data where you can set 
%      the position of the read data. dataStart is the position in data
%      where the first read datasample is stored.
%
%       Please note, that the fields chanidx and dataPos used as c indices 
%       starting at 0.
% RETURNS
%          data: [nChans, len] the actual data
%
% DESCRIPTION
%    Open a file and read the eeg data. The data is filtered with an IIR 
%    filter and an FIR filter.
%
%    For the eeg file we assume that it was written with the following
%    settings: DataFormat = BINARY
%              DataOrientation = MULTIPLEXED
%              BinaryFormat = INT_16
%
% COMPILE WITH
%    mex read_bv.c
%
% AUTHOR
%    Max Sagebaum
%
%    2008/04/15 - Max Sagebaum
%                   - file created 
% (c) 2005 Fraunhofer FIRST