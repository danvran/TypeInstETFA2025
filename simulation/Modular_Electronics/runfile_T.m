% Saves files in the format
% model_model-type_anomaly-type_mean-sine-offset.csv
% model-type: simulation model used
% anomaly-type: defects within modules
% mean-sine-offset Offset of the input source voltage

clear; % Clear workspace
varList = ["anomalies", "test", "train"];
varList = ["anomalies"];

for k = 1:2
    % Create the model name using strcat
    model = strcat('RLC_T', num2str(k));  % Using num2str to convert i to string

    for j = 1:length(varList)
        % Get the current variable name
        currentVar = varList{j};
        clearvars -except currentVar i varList model k
        
        switch currentVar
            case "anomalies"  % This must run after the "variables" script
                disp('Running Anomalies');
                variables;  % Run script to initialize all model variables
                % Specify the directory path
                anomalyPath = './anomalies_T';
                anomalyFiles = dir(anomalyPath);
                for f = 1:length(anomalyFiles)
                    if anomalyFiles(f).isdir == 0 % Assure it's not a directory
                        filename = anomalyFiles(f).name;
                        disp(['Processing file: ' filename]);
                        variables;  % Run script to initialize all model variables
                        runtime = anomaly_runtime;  %
                        scriptPath = fullfile(anomalyPath, filename);
                        run(scriptPath);
                        for i = 0:2:20
                            % Update the value of sine_offset
                            sine_offset.Value = i;
                            % Call the function with updated sine_offset
                            clean_name = erase(filename, '.m');
                            directory = sprintf('./data/model_T%d/anom', k);
                            csvname = sprintf('./data/model_T%d/anom/RLC_T%d_%d_%s.csv', k, k, i, clean_name);
                            processFile(model, directory, csvname, currentVar)
                        end
                    end
                end
            case "test"
                disp('Running Test');
                variables;  % Run script to initialize all model variables
                runtime = anomaly_runtime;
                for i = 0:2:20
                    % Update the value of sine_offset
                    sine_offset.Value = i;
                    % Call the function with updated sine_offset
                    directory = sprintf('./data/model_T%d/test', k);
                    csvname = sprintf('./data/model_T%d/test/RLC_T%d_%d_OK.csv', k, k, i);
                    processFile(model, directory, csvname, currentVar)
                end
            case "train"
                disp('Running Train');
                variables;  % Run script to initialize all model variables
                for i = 0:2:20
                    sine_offset.Value = i;
                    directory = sprintf('./data/model_T%d/train', k);
                    csvname = sprintf('./data/model_T%d/train/RLC_T%d_%d_OK.csv', k, k, i);
                    processFile(model, directory, csvname, currentVar)
                end
        end
    end
end
    
    % Function to process each file
    function processFile(model, directory, csvname, currentVar)
        simOut = sim(model);
        V_T1_I1 = squeeze(simOut.V_T1_I1);
        I_T1_I1 = squeeze(simOut.I_T1_I1);
        
        V_T1_I1 = squeeze(V_T1_I1);
        I_T1_I1 = squeeze(I_T1_I1);
    
        T = table(V_T1_I1, ...
            I_T1_I1);
    
        switch currentVar
            case "anomalies"
                T(1:100,:) = []; % remove first 100 sample times
            case "test"
                T(1:100,:) = []; % remove first 100 sample times
            case "train"
                T(1:100100,:) = []; % remove first 1001 sample times
            otherwise
                disp('Unknown variable.');
        end
        if ~isempty(directory) && ~exist(directory, 'dir')
            mkdir(directory);  % Creates the directory and any missing intermediate directories
        end
        writetable(T, csvname)
    end