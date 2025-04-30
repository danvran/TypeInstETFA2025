% Use Simulink.Parameters to populate workspace variables for Simulink
% Run this script to initialize all normal behavior variables
% Then run an anomaly script or multiple ones to generate anomalies

% File export path
csvname = './data/RLC.csv';

% Gloabal Variables
runtime = Simulink.Parameter(6001.0);
anomaly_runtime = Simulink.Parameter(1001.0);
samplefrequency = 100.0;
sampletime = Simulink.Parameter(1/samplefrequency);  % T_s and T_0
sine_offset = Simulink.Parameter(0.0);
sine_amp = Simulink.Parameter(0.5);
sine_freq = Simulink.Parameter(10.0);
resistance_T1 = Simulink.Parameter(1.0);
inductance_T1 = Simulink.Parameter(0.5);
capacitance_T1 = Simulink.Parameter(0.1);