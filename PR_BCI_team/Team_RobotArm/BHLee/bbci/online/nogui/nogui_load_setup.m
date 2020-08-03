function setup= nogui_load_setup(file)

global BCI_DIR general_port_fields

if ~isabsolutepath(file),
  file= strcat(BCI_DIR, 'online/setups/', file);
end

fid = fopen(file,'r');
F = fread(fid);
eval(char(F'));

setup.general.active1= 1;
setup.general.setup_directory= [BCI_DIR 'online/setups/'];
setup.graphic_player1.machine= general_port_fields.graphic{1};
setup.graphic_player1.port= general_port_fields.graphic{2};
setup.graphic_player1.fb_port= general_port_fields.control{3};
setup.control_player1.machine= general_port_fields.control{1};
setup.control_player1.port= general_port_fields.control{2};

if isfield(setup.graphic_player1.feedback_opt, 'directions') & ...
    isempty(setup.graphic_player1.feedback_opt.directions{1}),
  setup.graphic_player1.feedback_opt.directions= setup.graphic_player1.feedback_opt.classes;
end

% this should not be neccessary
setup.gui_machine= general_port_fields.bvmachine;
setup.graphic_player1.gui_machine= general_port_fields.bvmachine;
setup.control_player1.gui_machine= general_port_fields.bvmachine;
