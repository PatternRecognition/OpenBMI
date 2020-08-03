function bbci_save_otherPics(bbci,data)

tmp=[data.all_results(1).figure_handles,data.all_results(2).figure_handles,data.all_results(3).figure_handles];
others=setdiff(tmp,data.figure_handles);
file= fullfile(bbci.calibrate.save.folder, bbci.calibrate.save.file);
fig_folder_others= strcat(file, '_other_figures');
 for ff= others(:)',
    figure(ff);
    fig_name= strrep(get(ff,'Name'), ' ', '_');
    if ispc,
      fig_name= strrep(fig_name, '.', '_');
    end
    filename= fullfile(fig_folder_others, sprintf('Fig-%02d_%s', ff, fig_name));
    printFigure(filename, bbci.calibrate.save.figures_spec{:});
    set(figure(ff), 'Visible','off');
 end
