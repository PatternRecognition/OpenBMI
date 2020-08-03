function mexSSClient
% mexSSClient - get data from the signalserver
%
% SYNOPSIS
%    [master_info sig_info ch_info] = mexSSClient(host,port, protocol)  [init]
%    mexSSClient('close')                                              [close]
%    [info,data] = mexSSClient()                                    [get data]
%
% ARGUMENTS
%              host: The name of the machine where the signalserver runs.
%              port: The port at which the signal server listens for the
%                    connection.
%          protocol: The protocol you want to use for data transmission
%                    choose either 'tcp' or 'udp'.
%
% RETURNS
%   master_info: Array with the information for the master signal 
%                [master_freq block_size]
%                master_freq: The sampling frequency of the master signal
%                 block_size: The number of samples in each data block
%      sig_info: Cellarray which contains the information for each signal
%                type. For each signal type you have the following values:
%                {signal_number signal_name block_size nr_of_channels freq}
%                 signal_number: The number for the signal type.
%                   signal_name: The name for the signal (e.g. eeg,emg,...)
%                    block_size: The number of samples in each data block.
%                          freq: The frequency for the signal.
%       ch_info: Cellarray with the name and the type of each channel. For
%                each channel you have {chan_name, signal_type}
%                   chan_name: Name of the channel.
%                 signal_type: Name of the signal type.
%          info: A structure with general information about the recieved
%                package. Info contains the following fields:
%               .sampleNr: The sample number of the package.
%          data: A cellarray with the signal data. The array has a row for 
%                every signal type. The row contains the fields
%                {signal_name data}
%                 signal_name: The name of the signal type.
%                        data: The data which was in the package. The
%                        matrix has the size 'samples'x'channel_number'.
%
% DESCRIPTION
%     With the mexSSClient you can connect to a running signalserver. Once
%    connected to the server you will receive the data with additional
%    calls to mexSSClient.
%     The first call returns a structure with the configuration details of
%    the signal server. It contains information about the master frequency
%    and block size. You get information about the properties of each signal
%    type and a list which contains the information about all channels
%    recorded by the singalserver.
%
%     If you want to connect to a signal server which runs on the machine
%    10.10.10.1 listening to the port 9000 you simply need to call:
%
%        [master_info sig_info ch_info] = mexSSClient('10.10.10.1',9000,'tcp')
%
%      Now the client is ready to receive packages from the server. You can
%      get the next package with the call:
%
%        [info,data] = mexSSClient()
%
%       And you will have the data of the next package available. If there is
%      no package from the server the client will wait until it receives
%      the next package. So the call is blocking.
%       When you are finished with the signal acquisition you close the
%      connection to the server with:
%
%        mexSSClient('close')
%
% COMMENTS
%   This is a copy from the file in the signal server tobi project. If you
%   want to have the newest version, you have to check there.
%   Also the file binary files are copied from the tobi project.
%
% AUTHOR
%    Max Sagebaum 
%
%    2010/04/07 Max Sagebaum
%                    - file created