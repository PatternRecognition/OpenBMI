This two is demo ripped from a demo solotion for a real application.

The DAQ simulator:

As I do not have a DAQ system,  I made a simple DAQ simulator.
The script sim_daq.m simulates a daq system and sends some data packets with sinus wave data in UDP packets.
Start it in its own MATLAB window on the same computer or an other one start it with for an example:

sim_daq('localhost',3001,3002)

It will send 16-bit data with "Network" byte order. You may change it to "Intel" byte order if thats more correct.
If transmisstion speed it to fast. then will the network or reciving application loss packets.
You can adjust delay with the pause(xxx) comands. you will read out different average transmission speeds from this.
I fill the initial 2-byte position with 16-bit "id" number for the transmission.


DAQ Reciver:

First of all I switched over to use only pnet comunication and no matlabs own UDP.
You can run it like this:

[data_matrix,id_list]=new_transmit_and_get('localhost',3002,3001);
plot(data_matrix);

You will se the sinus waves plotted if you receive from the sim_daq application.
You may change the byte order if im wrong about  the byteorder your daq system use....