%proc_detEntropy calculates an entropy value for the given data.
%
% description: 
%  We want to calculate the deterministic entropy (H), a complexity measure (C) 
%  and an information measure (I) value for a time series x. For each
%  scanline received in the function call, we create a
%  binary string like '0010011110011' in the following way: 
%  Char at position i is 0 if the x(i) is less than the
%  position of the scan line. Otherwise it is 1.
%
% For the entropy calculation we use the qtc method from Mark Titchener.
%
% usage:
%     [H,C,I] = proc_detEntropy(x,scanLinePos, winSize,'StepSize',stepSize);
%     [H,C,I] = proc_detEntropy(x,scanLinePos, winSize,'WinPos', winPos);
%     [h, c, i] = proc_detEntropy(x,scanLinePos, winSize,'StepSize',stepSize, 'Concat', 1);
%     [h, c, i] = proc_detEntropy(x,scanLinePos, winSize,'WinPos', winPos, 'Concat', 1);
%     
%     Each window position results in a column of H or h. Each scanline position
%     results in column of H. Given a set of n window positions and a set of m scan
%     line positions H will look like this:
%         | winPos_1:scanLinePos_1 winPos_2:scanLinePos_1 ... winPos_n:scanLinePos_1 | 
%         | winPos_1:scanLinePos_2 winPos_2:scanLinePos_2 ... winPos_n:scanLinePos_2 |
%                    .                     .                       .
%    H =             .                     .                       .
%                    .                     .                       .
%         | winPos_1:scanLinePos_m winPos_2:scanLinePos_m ... winPos_n:scanLinePos_m |
%
%     respectively for C and I   
%
%     The second two function calls will do the same but it calculates
%     only one value for each window position by concatenating the
%     bitstrings. Given a set of n window positions h will look
%     like:
%
%    h = |winPos_1, winPos_2, ... , winPos_n|
%
%     respectively for c and i   
%
% input:
%     x               - [1:n] the time series (float values)
%     scanLinePos     - [1:m] the positions of the scanlines
%     winSize         - the size in samples of the windows to analyse
%     stepSize        - if you set this option the windows will start every
%                     step size points until the end of the data is reached
%     winPos          - [1:n] if you set this option the windows will start at the
%                     positions specified in winPos
%     'Concat',value  value:1 if you want to calculate a single value per window
%                     by chaining the bitstrings of the scan line positions
%                     (Default: value=0)
%
%    Note: You have to set one of the two variables winPos or stepSize.
%          specifying two or zero will result in an error.
%          'Concat' is an optional value.
%
% output:
%     H,h      - The matrix H or vector h with the entropy values
%                Size of H: [(number of scan lines) x (number of windows)]
%                Size of h: [1 x (number of windows)]
%     C,c      - The matrix C or vector c with the complexity values
%                Size? analog to H, h
%     I,i      - The matrix I or vector i with the information values
%                Size? analog to H, h
%
% AUTHOR
%    Max Sagebaum
%
%    2008/04/22 - Max Sagebaum
%                   - file created 
% (c) 2005 Fraunhofer FIRST

