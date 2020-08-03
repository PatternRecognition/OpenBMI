mnt= getElectrodePositions(clab);
drawScalpOutline(mnt, ...
                 'showLabels', 1);

printFigure('/tmp/electrode_layout', [15 16], 'format','pdf');

                 