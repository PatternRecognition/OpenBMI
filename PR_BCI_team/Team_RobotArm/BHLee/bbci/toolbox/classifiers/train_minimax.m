function r= train_minimax(data, labels, varargin)
%TRAINMINIMAX implement MiniMax
% The algorithm initialisizes the MiniMax Probability machine
% Input: data: a set of observed data written in columns
%        labels: a logical label array
%        opt: struct, with .maxnumber and .fault value for the iteration.
% Output: A trained Minimax for a two-point-classfier w'z=-b
%         r is a structure:
%           r.w : w from w'z=-b
%           r.b : b from w'z=-b
%           r.alpha = expected likelihood for right classfication of future datas
%
% Guido Dornhege
% 09.01.02

r = train_Rminimax(data, labels, 0, 0, varargin);
