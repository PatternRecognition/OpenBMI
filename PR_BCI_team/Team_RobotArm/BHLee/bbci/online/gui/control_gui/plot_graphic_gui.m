function plot_graphic_gui(fig,player);
%PLOT_GRAPHIC_GUI ONLY FOR INTERN USE OF MATLAB_CONTROL_GUI
%
% plots the graphic part of the gui
% 
% usage:
%    plot_graphic_gui(fig,player);
%
% input:
%    fig    the handle of the gui
%    player the player number
%
% Guido Dornhege
% $Id: plot_graphic_gui.m,v 1.1 2006/04/27 14:21:11 neuro_cvs Exp $

% CALL CONTROL GUI AS LONG AS THEY DO NEARBY THE SAME
plot_control_gui(fig,player,'graphic');

